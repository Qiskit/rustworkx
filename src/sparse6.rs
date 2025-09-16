use pyo3::prelude::*;
use pyo3::types::PyAny;
use crate::graph6::IOError;
use crate::graph::PyGraph;
use petgraph::graph::NodeIndex;
use petgraph::prelude::Undirected;
use crate::StablePyGraph;
use std::iter;

fn parse_n(bytes: &[u8], pos: usize) -> Result<(usize, usize), IOError> {
    if pos >= bytes.len() {
        return Err(IOError::NonCanonicalEncoding);
    }
    let first = bytes[pos];
    if first < 63 || first > 126 {
        return Err(IOError::InvalidSizeChar);
    }
    if first != 126 {
        return Ok(((first - 63) as usize, pos + 1));
    }
    // first == 126 -> extended form. Look ahead without advancing permanently.
    if pos + 1 >= bytes.len() {
        return Err(IOError::NonCanonicalEncoding);
    }
    let second = bytes[pos + 1];
    if second == 126 {
        // 8 byte form: 126 126 R(x) where R(x) is 36 bits -> 6 bytes
        if bytes.len() < pos + 2 + 6 {
            return Err(IOError::NonCanonicalEncoding);
        }
        let mut val: u64 = 0;
        for i in 0..6 {
            let b = bytes[pos + 2 + i];
            if b < 63 || b > 126 {
                return Err(IOError::InvalidSizeChar);
            }
            val = (val << 6) | ((b - 63) as u64);
        }
        return Ok((val as usize, pos + 2 + 6));
    } else {
        // 4 byte form: 126 R(x) where R(x) is 18 bits -> 3 bytes
        if bytes.len() < pos + 1 + 3 {
            return Err(IOError::NonCanonicalEncoding);
        }
        let mut val: usize = 0;
        for i in 0..3 {
            let b = bytes[pos + 1 + i];
            if b < 63 || b > 126 {
                return Err(IOError::InvalidSizeChar);
            }
            val = (val << 6) | ((b - 63) as usize);
        }
        if val == 64032 {
            eprintln!("DEBUG sparse6 parse_n anomaly: pos={} bytes_prefix={:?} triple={:?}", pos, &bytes[..std::cmp::min(bytes.len(),10)], [&bytes[pos+1], &bytes[pos+2], &bytes[pos+3]]);
        }
        return Ok((val, pos + 1 + 3));
    }
}

fn bits_from_bytes(bytes: &[u8], start: usize) -> Result<Vec<u8>, IOError> {
    let mut bits = Vec::new();
    for &b in bytes.iter().skip(start) {
        if b < 63 || b > 126 {
            return Err(IOError::InvalidSizeChar);
        }
        let val = b - 63;
        for i in 0..6 {
            let bit = (val >> (5 - i)) & 1;
            bits.push(bit);
        }
    }
    Ok(bits)
}

// Encoder: produce sparse6 byte chars (63-based) from a graph's bit_vec
fn to_sparse6_bytes(bit_vec: &[usize], n: usize, header: bool) -> Result<Vec<u8>, IOError> {
    if n >= (1usize << 36) {
        return Err(IOError::GraphTooLarge);
    }
    let mut out: Vec<u8> = Vec::new();
    if header {
        out.extend_from_slice(b">>sparse6<<");
    }
    out.push(b':');

    // write N(n) using same encoding as graph6 utils.get_size but extended
    if n < 63 {
        out.push((n as u8) + 63);
    } else if n < (1 << 18) {
        // 4-byte form: 126 then three 6-bit chars
        out.push(126);
        let mut val = n as usize;
        let mut parts = [0u8; 3];
        parts[2] = (val & 0x3F) as u8;
        val >>= 6;
        parts[1] = (val & 0x3F) as u8;
        val >>= 6;
        parts[0] = (val & 0x3F) as u8;
        for p in parts.iter() {
            out.push(p + 63);
        }
    } else {
        // 8-byte form: 126,126 then 6-byte 36-bit value
        out.push(126);
        out.push(126);
        let mut val = n as u64;
        let mut parts = [0u8; 6];
        for i in (0..6).rev() {
            parts[i] = (val as u8) & 0x3F;
            val >>= 6;
        }
        for p in parts.iter() {
            out.push(p + 63);
        }
    }

    // compute k
    let mut k = 1usize;
    while (1usize << k) < n {
        k += 1;
    }

    // Build edges from bit_vec
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for i in 0..n {
        for j in 0..=i {
            if bit_vec[i * n + j] == 1 {
                edges.push((i, j));
            }
        }
    }
    // edges should be sorted by (v=max, u=min)
    edges.sort_by_key(|(a, b)| (*a, *b));

    let mut bits: Vec<u8> = Vec::new();
    let mut curv = 0usize;
    for (v, u) in edges.iter() {
        let v = *v;
        let u = *u;
        if v == curv {
            bits.push(0);
            for i in (0..k).rev() {
                bits.push(((u >> i) & 1) as u8);
            }
        } else if v == curv + 1 {
            curv += 1;
            bits.push(1);
            for i in (0..k).rev() {
                bits.push(((u >> i) & 1) as u8);
            }
        } else {
            curv = v;
            bits.push(1);
            for i in (0..k).rev() {
                bits.push(((v >> i) & 1) as u8);
            }
            bits.push(0);
            for i in (0..k).rev() {
                bits.push(((u >> i) & 1) as u8);
            }
        }
    }

    // padding: canonical calculation
    let pad = (6 - (bits.len() % 6)) % 6;
    if k < 6 && n == (1 << k) && pad >= k && curv < (n - 1) {
        // special-case: prepend a 0 then pad with 1s
        bits.push(0);
    }
    bits.extend(iter::repeat(1u8).take(pad));

    // pack into 6-bit chars
    for chunk in bits.chunks(6) {
        let mut val = 0u8;
        for b in chunk.iter() {
            val = (val << 1) | (b & 1);
        }
        out.push(val + 63);
    }
    out.push(b'\n');
    Ok(out)
}

#[pyfunction]
#[pyo3(signature=(pygraph, header=true))]
pub fn write_sparse6_from_pygraph(pygraph: Py<PyGraph>, header: bool) -> PyResult<String> {
    Python::with_gil(|py| {
        let g = pygraph.borrow(py);
        let n = g.graph.node_count();
        let mut bit_vec = vec![0usize; n * n];
        for (i, j, _w) in crate::get_edge_iter_with_weights(&g.graph) {
            bit_vec[i * n + j] = 1;
            bit_vec[j * n + i] = 1;
        }
        let bytes = to_sparse6_bytes(&bit_vec, n, header)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("sparse6 encode error: {:?}", e)))?;
        // convert bytes to string
        let s = String::from_utf8(bytes).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("utf8: {}", e)))?;
        Ok(s)
    })
}

#[pyfunction]
#[pyo3(signature=(repr))]
pub fn read_sparse6_str<'py>(py: Python<'py>, repr: &str) -> PyResult<Bound<'py, PyAny>> {
    let s_trim = repr.trim_end_matches('\n');
    if s_trim.is_empty() { return Err(PyErr::from(IOError::NonCanonicalEncoding)); }

    let wrapped = std::panic::catch_unwind(|| {
        // Accept optional leading ':' or ';' for incremental form
        let mut s = s_trim.as_bytes();
    if s.starts_with(b">>sparse6<<:") { s = &s[12..]; }
        let mut pos = 0usize;
        if s.len() > 0 && (s[0] == b';' || s[0] == b':') {
            pos = 1;
        }

        // Parse N(n)
    let (n, new_pos) = parse_n(s, pos)?;
    let pos = new_pos;

        // compute k = bits needed to represent n-1
        let k = if n <= 1 { 0 } else { (usize::BITS - (n - 1).leading_zeros()) as usize };
        if pos >= s.len() {
            return Ok::<(Vec<(usize, usize)>, usize), IOError>((Vec::new(), n));
        }
        let bits = bits_from_bytes(s, pos)?;
        let mut idx = 0usize;
        let mut v: usize = 0;
        let mut edges: Vec<(usize, usize)> = Vec::new();
        while idx + 1 + k <= bits.len() {
            let b = bits[idx];
            idx += 1;
            let mut x: usize = 0;
            for _ in 0..k { x = (x << 1) | (bits[idx] as usize); idx += 1; }
            if b == 1 { v = v.saturating_add(1); }
            if x > v { v = x; }
            else if x < v && x < n && v < n { edges.push((x, v)); }
            if idx < bits.len() && bits[idx..].iter().all(|&b| b == 1) { break; }
        }
        Ok((edges, n))
    });

    match wrapped {
    Ok(Ok((edges, n))) => {
            // convert to PyGraph
            let mut graph = StablePyGraph::<Undirected>::with_capacity(n, 0);
            for _ in 0..n {
                graph.add_node(py.None());
            }
            for (u, v) in edges { graph.add_edge(NodeIndex::new(u), NodeIndex::new(v), py.None()); }
            let out = PyGraph { graph, node_removed: false, multigraph: true, attrs: py.None() };
            Ok(out.into_pyobject(py)?.into_any())
        }
        Ok(Err(io_err)) => Err(PyErr::from(io_err)),
        Err(panic_payload) => {
            let msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                format!("Rust panic in sparse6 parser: {}", s)
            } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                format!("Rust panic in sparse6 parser: {}", s)
            } else {
                "Rust panic in sparse6 parser (non-string payload)".to_string()
            };
            Err(crate::Graph6PanicError::new_err(msg))
        }
    }
}

// NOTE: 4-byte form currently misinterprets because spec is 126 + three 6-bit chars, not including the second byte in prior logic.
