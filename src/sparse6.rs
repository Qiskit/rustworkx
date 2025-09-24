use crate::graph::PyGraph;
use crate::graph6::{GraphNumberCodec, IOError, SizeCodec};
use crate::StablePyGraph;
use petgraph::graph::NodeIndex;
use petgraph::prelude::Undirected;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::iter;

/// Parse an n value from a sparse6 stream using the shared GraphNumberCodec.
/// Returns (n, absolute next byte position). Enforces canonical size encoding
/// and raises GraphTooLarge if n >= 2^36.
fn parse_n(bytes: &[u8], pos: usize) -> Result<(usize, usize), IOError> {
    let (n, consumed) = GraphNumberCodec::decode_size(bytes, pos)?;
    Ok((n, pos + consumed))
}

/// Encode an undirected graph adjacency matrix (bit_vec of length n*n) into
/// sparse6 bytes. If `header` is true the ">>sparse6<<" marker is prepended.
/// Applies canonical padding rules and returns a newline terminated buffer.
fn to_sparse6_bytes(bit_vec: &[usize], n: usize, header: bool) -> Result<Vec<u8>, IOError> {
    // Unified bound check occurs inside GraphNumberCodec::encode_size too, but keep for clarity.
    if n >= (1usize << 36) {
        return Err(IOError::GraphTooLarge);
    }
    let mut out: Vec<u8> = Vec::new();
    if header {
        out.extend_from_slice(b">>sparse6<<");
    }
    out.push(b':');
    let size_enc = GraphNumberCodec::encode_size(n)?;
    out.extend_from_slice(&size_enc);

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
/// Encode a `PyGraph` to sparse6 format and return the ASCII string. When
/// `header` is true the standard ">>sparse6<<:" header is included. Fails on
/// non‑canonical or oversized graphs (n >= 2^36). Ignores parallel edges.
pub fn graph_write_sparse6_to_str<'py>(
    py: Python<'py>,
    pygraph: Py<PyGraph>,
    header: bool,
) -> PyResult<String> {
    let g = pygraph.borrow(py);
    let n = g.graph.node_count();
    let mut bit_vec = vec![0usize; n * n];
    for (i, j, _w) in crate::get_edge_iter_with_weights(&g.graph) {
        bit_vec[i * n + j] = 1;
        bit_vec[j * n + i] = 1;
    }
    let bytes = to_sparse6_bytes(&bit_vec, n, header).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("sparse6 encode error: {:?}", e))
    })?;
    // convert bytes to string
    let s = String::from_utf8(bytes)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("utf8: {}", e)))?;
    Ok(s)
}

#[pyfunction]
#[pyo3(signature=(repr))]
/// Parse a sparse6 string (optionally containing the standard header) into a
/// `PyGraph`. Accepts trailing newline, tolerates leading ':' or ';'. Performs
/// bounds and character validation and converts Rust panics into Python errors.
pub fn read_sparse6_str<'py>(py: Python<'py>, repr: &str) -> PyResult<Bound<'py, PyAny>> {
    let s_trim = repr.trim_end_matches('\n');
    if s_trim.is_empty() {
        return Err(PyErr::from(IOError::NonCanonicalEncoding));
    }

    let wrapped = std::panic::catch_unwind(|| {
        // Accept optional leading ':' or ';' for incremental form
        let mut s = s_trim.as_bytes();
        if s.starts_with(b">>sparse6<<:") {
            s = &s[12..];
        }
        let mut pos = 0usize;
        if s.len() > 0 && (s[0] == b';' || s[0] == b':') {
            pos = 1;
        }

        // Parse N(n) (returns absolute next index)
        let (n, pos_after) = parse_n(s, pos)?;
        let pos = pos_after;

        // compute k = bits needed to represent n-1
        let k = if n <= 1 {
            0
        } else {
            (usize::BITS - (n - 1).leading_zeros()) as usize
        };
        if pos >= s.len() {
            return Ok::<(Vec<(usize, usize)>, usize), IOError>((Vec::new(), n));
        }
        // let bits = bits_from_bytes(s, pos)?;
        let mut bits = Vec::new();
        for &b in s.iter().skip(pos) {
            if b < 63 || b > 126 {
                return Err(IOError::InvalidSizeChar);
            }
            let val = b - 63;
            for i in 0..6 {
                let bit = (val >> (5 - i)) & 1;
                bits.push(bit);
            }
        }
        let mut idx = 0usize;
        let mut v: usize = 0;
        let mut edges: Vec<(usize, usize)> = Vec::new();
        while idx + 1 + k <= bits.len() {
            let b = bits[idx];
            idx += 1;
            let mut x: usize = 0;
            for _ in 0..k {
                x = (x << 1) | (bits[idx] as usize);
                idx += 1;
            }
            if b == 1 {
                v = v.saturating_add(1);
            }
            if x > v {
                v = x;
            } else if x < v && x < n && v < n {
                edges.push((x, v));
            }
            if idx < bits.len() && bits[idx..].iter().all(|&b| b == 1) {
                break;
            }
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
            for (u, v) in edges {
                graph.add_edge(NodeIndex::new(u), NodeIndex::new(v), py.None());
            }
            let out = PyGraph {
                graph,
                node_removed: false,
                multigraph: true,
                attrs: py.None(),
            };
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
