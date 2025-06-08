# Binary Weight Format Specification

## Overview

This document specifies the binary format for storing model weights and vocabulary for the `cpp_embedder` library. The format is designed for efficient memory mapping and minimal parsing overhead.

## File Extension

`.weights` (recommended)

## Byte Order

All multi-byte values are stored in **little-endian** byte order.

---

## File Structure

```
+---------------------------+
|      File Header (64B)    |
+---------------------------+
|   Metadata Section        |
+---------------------------+
|   Vocabulary Section      |
+---------------------------+
|   Tensor Index (TOC)      |
+---------------------------+
|   Tensor Data             |
+---------------------------+
|   Footer (16B)            |
+---------------------------+
```

---

## File Header (64 bytes)

| Offset | Size | Type    | Field                | Description                          |
|--------|------|---------|----------------------|--------------------------------------|
| 0      | 4    | char[4] | magic                | Magic bytes: `0x45 0x4D 0x42 0x44` ("EMBD") |
| 4      | 2    | uint16  | version_major        | Format major version (currently 1)   |
| 6      | 2    | uint16  | version_minor        | Format minor version (currently 0)   |
| 8      | 4    | uint32  | flags                | Feature flags (see below)            |
| 12     | 4    | uint32  | metadata_offset      | Offset to metadata section           |
| 16     | 4    | uint32  | metadata_size        | Size of metadata section in bytes    |
| 20     | 4    | uint32  | vocab_offset         | Offset to vocabulary section         |
| 24     | 4    | uint32  | vocab_size           | Size of vocabulary section in bytes  |
| 28     | 4    | uint32  | tensor_index_offset  | Offset to tensor index               |
| 32     | 4    | uint32  | tensor_index_count   | Number of tensors in index           |
| 36     | 4    | uint32  | tensor_data_offset   | Offset to tensor data                |
| 40     | 8    | uint64  | tensor_data_size     | Total size of tensor data in bytes   |
| 48     | 8    | uint64  | total_file_size      | Total file size for validation       |
| 56     | 4    | uint32  | header_checksum      | CRC32 of bytes 0-55                  |
| 60     | 4    | uint32  | reserved             | Reserved for future use (set to 0)   |

### Flags Field (Bit Definitions)

| Bit | Name                | Description                              |
|-----|---------------------|------------------------------------------|
| 0   | VOCAB_EMBEDDED      | Vocabulary is embedded in file           |
| 1   | TENSORS_ALIGNED     | Tensor data is 64-byte aligned           |
| 2   | CHECKSUM_ENABLED    | File includes checksums                  |
| 3   | COMPRESSED          | Tensor data is compressed (reserved)     |
| 4-31| Reserved            | Set to 0                                 |

---

## Metadata Section

Variable-length section containing model metadata as key-value pairs.

### Metadata Header

| Offset | Size | Type   | Field           | Description                     |
|--------|------|--------|-----------------|---------------------------------|
| 0      | 4    | uint32 | entry_count     | Number of metadata entries      |
| 4      | 4    | uint32 | total_size      | Total size of all entries       |

### Metadata Entry (repeated)

| Offset | Size | Type     | Field        | Description                      |
|--------|------|----------|--------------|----------------------------------|
| 0      | 2    | uint16   | key_length   | Length of key string (UTF-8)     |
| 2      | 2    | uint16   | value_length | Length of value string (UTF-8)   |
| 4      | N    | char[N]  | key          | Key string (not null-terminated) |
| 4+N    | M    | char[M]  | value        | Value string (not null-terminated)|

### Required Metadata Keys

| Key                  | Example Value           | Description                        |
|----------------------|-------------------------|------------------------------------|
| `model_name`         | `all-MiniLM-L6-v2`     | Model identifier                   |
| `model_version`      | `1.0.0`                 | Model version                      |
| `embedding_dim`      | `384`                   | Output embedding dimension         |
| `vocab_size`         | `30522`                 | Vocabulary size                    |
| `num_layers`         | `6`                     | Number of transformer layers       |
| `num_attention_heads`| `12`                    | Number of attention heads          |
| `hidden_size`        | `384`                   | Hidden dimension                   |
| `intermediate_size`  | `1536`                  | FFN intermediate dimension         |
| `max_position_emb`   | `512`                   | Maximum position embeddings        |
| `created_at`         | `2025-01-16T12:00:00Z` | Creation timestamp (ISO 8601)      |

---

## Vocabulary Section

Contains the WordPiece vocabulary as a serialized token list.

### Vocabulary Header

| Offset | Size | Type   | Field           | Description                     |
|--------|------|--------|-----------------|---------------------------------|
| 0      | 4    | uint32 | token_count     | Number of tokens in vocabulary  |
| 4      | 4    | uint32 | total_size      | Total size of token data        |
| 8      | 4    | uint32 | special_tokens  | Offset to special token indices |

### Special Token Indices (at special_tokens offset)

| Offset | Size | Type   | Field        | Description               |
|--------|------|--------|--------------|---------------------------|
| 0      | 4    | uint32 | pad_id       | [PAD] token ID (usually 0)|
| 4      | 4    | uint32 | unk_id       | [UNK] token ID (usually 100)|
| 8      | 4    | uint32 | cls_id       | [CLS] token ID (usually 101)|
| 12     | 4    | uint32 | sep_id       | [SEP] token ID (usually 102)|
| 16     | 4    | uint32 | mask_id      | [MASK] token ID (usually 103)|

### Token Entries (repeated, indexed by ID)

Tokens are stored sequentially, with token ID being the implicit index (0, 1, 2, ...).

| Offset | Size | Type     | Field        | Description                       |
|--------|------|----------|--------------|-----------------------------------|
| 0      | 2    | uint16   | token_length | Length of token string (UTF-8)    |
| 2      | N    | char[N]  | token        | Token string (not null-terminated)|

**Note**: Tokens with WordPiece continuation prefix use `##` as stored in the original vocabulary.

---

## Tensor Index (Table of Contents)

Array of tensor descriptors providing metadata and location of each tensor.

### Tensor Descriptor (32 bytes each)

| Offset | Size | Type     | Field          | Description                          |
|--------|------|----------|----------------|--------------------------------------|
| 0      | 4    | uint32   | name_hash      | FNV-1a hash of tensor name           |
| 4      | 1    | uint8    | dtype          | Data type (see below)                |
| 5      | 1    | uint8    | ndim           | Number of dimensions (1-4)           |
| 6      | 2    | uint16   | name_length    | Length of name string                |
| 8      | 4    | uint32   | shape[0]       | Dimension 0 size                     |
| 12     | 4    | uint32   | shape[1]       | Dimension 1 size (0 if ndim < 2)     |
| 16     | 4    | uint32   | shape[2]       | Dimension 2 size (0 if ndim < 3)     |
| 20     | 4    | uint32   | shape[3]       | Dimension 3 size (0 if ndim < 4)     |
| 24     | 8    | uint64   | data_offset    | Offset from tensor_data_offset       |

### Data Types

| Value | Name      | Size (bytes) | Description              |
|-------|-----------|--------------|--------------------------|
| 0     | FLOAT32   | 4            | IEEE 754 single precision|
| 1     | FLOAT16   | 2            | IEEE 754 half precision  |
| 2     | BFLOAT16  | 2            | Brain floating point     |
| 3     | INT32     | 4            | Signed 32-bit integer    |
| 4     | INT16     | 2            | Signed 16-bit integer    |
| 5     | INT8      | 1            | Signed 8-bit integer     |
| 6     | UINT32    | 4            | Unsigned 32-bit integer  |
| 7     | UINT16    | 2            | Unsigned 16-bit integer  |
| 8     | UINT8     | 1            | Unsigned 8-bit integer   |

### Tensor Name String Table

Immediately following the tensor descriptors is a string table containing tensor names.

| Offset | Size | Type     | Field | Description                        |
|--------|------|----------|-------|------------------------------------|
| 0      | N    | char[N]  | names | Concatenated tensor name strings   |

Names are stored in descriptor order, lengths specified in each descriptor.

---

## Tensor Data Section

Raw tensor data stored contiguously. When `TENSORS_ALIGNED` flag is set, each tensor starts at a 64-byte aligned offset (padding bytes are zeros).

### Data Layout

Tensors are stored in **row-major (C-contiguous)** order:
- For a tensor of shape `[D0, D1, D2, D3]`, element at index `[i, j, k, l]` is at offset:
  `i * (D1 * D2 * D3) + j * (D2 * D3) + k * D3 + l`

---

## Required Tensors for all-MiniLM-L6-v2

### Embedding Tensors

| Tensor Name                               | Shape            | DType   | Description                |
|-------------------------------------------|------------------|---------|----------------------------|
| `embeddings.word_embeddings.weight`       | [30522, 384]     | FLOAT32 | Token embeddings           |
| `embeddings.position_embeddings.weight`   | [512, 384]       | FLOAT32 | Position embeddings        |
| `embeddings.token_type_embeddings.weight` | [2, 384]         | FLOAT32 | Segment embeddings         |
| `embeddings.LayerNorm.weight`             | [384]            | FLOAT32 | Embedding layer norm gamma |
| `embeddings.LayerNorm.bias`               | [384]            | FLOAT32 | Embedding layer norm beta  |

### Encoder Layers (repeated for layers 0-5)

Pattern: `encoder.layer.{N}.{component}` where N = 0, 1, 2, 3, 4, 5

| Tensor Name Suffix                         | Shape           | DType   | Description              |
|--------------------------------------------|-----------------|---------|--------------------------|
| `.attention.self.query.weight`             | [384, 384]      | FLOAT32 | Query projection weight  |
| `.attention.self.query.bias`               | [384]           | FLOAT32 | Query projection bias    |
| `.attention.self.key.weight`               | [384, 384]      | FLOAT32 | Key projection weight    |
| `.attention.self.key.bias`                 | [384]           | FLOAT32 | Key projection bias      |
| `.attention.self.value.weight`             | [384, 384]      | FLOAT32 | Value projection weight  |
| `.attention.self.value.bias`               | [384]           | FLOAT32 | Value projection bias    |
| `.attention.output.dense.weight`           | [384, 384]      | FLOAT32 | Attention output weight  |
| `.attention.output.dense.bias`             | [384]           | FLOAT32 | Attention output bias    |
| `.attention.output.LayerNorm.weight`       | [384]           | FLOAT32 | Attention LN gamma       |
| `.attention.output.LayerNorm.bias`         | [384]           | FLOAT32 | Attention LN beta        |
| `.intermediate.dense.weight`               | [1536, 384]     | FLOAT32 | FFN intermediate weight  |
| `.intermediate.dense.bias`                 | [1536]          | FLOAT32 | FFN intermediate bias    |
| `.output.dense.weight`                     | [384, 1536]     | FLOAT32 | FFN output weight        |
| `.output.dense.bias`                       | [384]           | FLOAT32 | FFN output bias          |
| `.output.LayerNorm.weight`                 | [384]           | FLOAT32 | FFN output LN gamma      |
| `.output.LayerNorm.bias`                   | [384]           | FLOAT32 | FFN output LN beta       |

### Pooler (Optional - not used for mean pooling)

| Tensor Name                | Shape        | DType   | Description              |
|----------------------------|--------------|---------|--------------------------|
| `pooler.dense.weight`      | [384, 384]   | FLOAT32 | Pooler projection weight |
| `pooler.dense.bias`        | [384]        | FLOAT32 | Pooler projection bias   |

---

## Footer (16 bytes)

| Offset | Size | Type     | Field          | Description                        |
|--------|------|----------|----------------|------------------------------------|
| 0      | 4    | uint32   | data_checksum  | CRC32 of tensor data section       |
| 4      | 4    | uint32   | file_checksum  | CRC32 of bytes 0 to (footer-1)     |
| 8      | 4    | char[4]  | magic_end      | End magic: `0x44 0x42 0x4D 0x45` ("DBME") |
| 12     | 4    | uint32   | reserved       | Reserved (set to 0)                |

---

## Checksum Algorithm

CRC32 using polynomial `0xEDB88320` (IEEE 802.3 / zlib compatible).

---

## Size Calculation

For all-MiniLM-L6-v2 with FLOAT32 weights:

| Component               | Calculation                          | Size (bytes)   |
|-------------------------|--------------------------------------|----------------|
| Word embeddings         | 30522 * 384 * 4                      | 46,897,664     |
| Position embeddings     | 512 * 384 * 4                        | 786,432        |
| Token type embeddings   | 2 * 384 * 4                          | 3,072          |
| Embedding LayerNorm     | 384 * 4 * 2                          | 3,072          |
| Per-layer weights       | (384*384*4 + 384*4) * 4 = QKV+Out    | 2,363,904      |
|                         | + (384*1536*4 + 1536*4) + (1536*384*4 + 384*4) | 4,722,688 |
|                         | + LayerNorm * 2 = 384 * 4 * 4        | 6,144          |
| All 6 layers            | 6 * (2,363,904 + 4,722,688 + 6,144)  | 42,556,416     |
| **Total tensor data**   |                                      | ~90 MB         |
| Vocabulary (~30K tokens)|                                      | ~300 KB        |
| **Total file size**     |                                      | ~91 MB         |

---

## Version History

| Version | Date       | Changes                              |
|---------|------------|--------------------------------------|
| 1.0     | 2025-01-16 | Initial specification                |

---

## Assumptions

1. Files are accessed via memory mapping when possible
2. Alignment padding uses zero bytes
3. String data is valid UTF-8
4. All offsets are absolute from file start unless noted
5. Hash collisions in tensor names are resolved by string comparison
