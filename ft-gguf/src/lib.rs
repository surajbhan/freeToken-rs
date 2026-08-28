//! ft-gguf: minimal GGUF v2/v3 reader — header, metadata KVs, tensor table,
//! mmap'd tensor data. Just enough to feed q4_0 MoE expert banks; not a
//! general GGUF toolkit.

use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Arr(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v as u64),
            Value::U16(v) => Some(v as u64),
            Value::U32(v) => Some(v as u64),
            Value::U64(v) => Some(v),
            Value::I8(v) if v >= 0 => Some(v as u64),
            Value::I16(v) if v >= 0 => Some(v as u64),
            Value::I32(v) if v >= 0 => Some(v as u64),
            Value::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// (elements per block, bytes per block) for the ggml tensor types we size.
fn block_layout(ggml_type: u32) -> Result<(usize, usize)> {
    Ok(match ggml_type {
        0 => (1, 4),     // f32
        1 => (1, 2),     // f16
        2 => (32, 18),   // q4_0
        3 => (32, 20),   // q4_1
        6 => (32, 22),   // q5_0
        7 => (32, 24),   // q5_1
        8 => (32, 34),   // q8_0
        10 => (256, 84),  // q2_k
        11 => (256, 110), // q3_k
        12 => (256, 144), // q4_k
        13 => (256, 176), // q5_k
        14 => (256, 210), // q6_k
        16 => (256, 66),  // iq2_xxs
        30 => (1, 2),    // bf16
        t => bail!("unsupported ggml type {t}"),
    })
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// ne[0] is the contiguous (row/k) dimension, ggml order.
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    /// byte offset within the data section
    pub offset: u64,
    pub nbytes: u64,
}

pub struct Gguf {
    mmap: Mmap,
    pub meta: HashMap<String, Value>,
    pub tensors: HashMap<String, TensorInfo>,
    data_start: usize,
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.buf.len() {
            bail!("gguf: truncated at {}+{n}", self.pos);
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8(self.take(n)?.to_vec())?)
    }
    fn value(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.take(1)?[0]),
            1 => Value::I8(self.take(1)?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.take(2)?.try_into()?)),
            3 => Value::I16(i16::from_le_bytes(self.take(2)?.try_into()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_le_bytes(self.take(4)?.try_into()?)),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let ety = self.u32()?;
                let n = self.u64()? as usize;
                let mut v = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    v.push(self.value(ety)?);
                }
                Value::Arr(v)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_le_bytes(self.take(8)?.try_into()?)),
            t => bail!("gguf: unknown value type {t}"),
        })
    }
}

impl Gguf {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("open {}", path.as_ref().display()))?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mut r = Reader { buf: &mmap, pos: 0 };

        if r.take(4)? != b"GGUF" {
            bail!("not a GGUF file");
        }
        let version = r.u32()?;
        if !(2..=3).contains(&version) {
            bail!("unsupported GGUF version {version}");
        }
        let n_tensors = r.u64()? as usize;
        let n_kv = r.u64()? as usize;

        let mut meta = HashMap::new();
        for _ in 0..n_kv {
            let key = r.string()?;
            let ty = r.u32()?;
            meta.insert(key, r.value(ty)?);
        }

        let mut tensors = HashMap::new();
        for _ in 0..n_tensors {
            let name = r.string()?;
            let n_dims = r.u32()? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(r.u64()?);
            }
            let ggml_type = r.u32()?;
            let offset = r.u64()?;
            let nelems: u64 = dims.iter().product();
            let (be, bb) = block_layout(ggml_type)?;
            if nelems as usize % be != 0 {
                bail!("tensor {name}: {nelems} elems not divisible by block {be}");
            }
            let nbytes = nelems / be as u64 * bb as u64;
            tensors.insert(name, TensorInfo { dims, ggml_type, offset, nbytes });
        }

        let alignment = meta
            .get("general.alignment")
            .and_then(Value::as_u64)
            .unwrap_or(32) as usize;
        let data_start = r.pos.div_ceil(alignment) * alignment;

        Ok(Self { mmap, meta, tensors, data_start })
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .get(name)
            .with_context(|| format!("tensor {name} not found"))
    }

    pub fn tensor_data(&self, name: &str) -> Result<&[u8]> {
        let t = self.tensor(name)?;
        let a = self.data_start + t.offset as usize;
        let b = a + t.nbytes as usize;
        if b > self.mmap.len() {
            bail!("tensor {name} data out of bounds");
        }
        Ok(&self.mmap[a..b])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Hand-craft a tiny GGUF v3 file: one KV, one q4_0 tensor.
    #[test]
    fn parses_handcrafted_file() -> Result<()> {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend(b"GGUF");
        buf.extend(3u32.to_le_bytes()); // version
        buf.extend(1u64.to_le_bytes()); // n_tensors
        buf.extend(1u64.to_le_bytes()); // n_kv
        // kv: "general.alignment" = u32 32
        let key = b"general.alignment";
        buf.extend((key.len() as u64).to_le_bytes());
        buf.extend(key);
        buf.extend(4u32.to_le_bytes()); // type u32
        buf.extend(32u32.to_le_bytes());
        // tensor: "t" dims [64, 2] q4_0 offset 0
        buf.extend(1u64.to_le_bytes());
        buf.extend(b"t");
        buf.extend(2u32.to_le_bytes());
        buf.extend(64u64.to_le_bytes());
        buf.extend(2u64.to_le_bytes());
        buf.extend(2u32.to_le_bytes()); // q4_0
        buf.extend(0u64.to_le_bytes());
        // pad to alignment, then data: 128 elems / 32 * 18 = 72 bytes
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        let data_start = buf.len();
        buf.extend(std::iter::repeat(0xABu8).take(72));

        let dir = std::env::temp_dir().join("ft-gguf-test");
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("tiny.gguf");
        File::create(&path)?.write_all(&buf)?;

        let g = Gguf::open(&path)?;
        assert_eq!(g.data_start, data_start);
        let t = g.tensor("t")?;
        assert_eq!(t.dims, vec![64, 2]);
        assert_eq!(t.nbytes, 72);
        assert!(g.tensor_data("t")?.iter().all(|&b| b == 0xAB));
        assert_eq!(g.meta["general.alignment"].as_u64(), Some(32));
        Ok(())
    }
}
