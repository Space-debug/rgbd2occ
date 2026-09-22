# -*- coding: utf-8 -*-
"""极简 MATLAB v7 MAT 文件解析器 (仅够读 SUNRGBDMeta.mat), 提取逐帧真实内参。
自 D:/Datasets/parse_sunrgbd_meta.py 移植, 未改解析逻辑。"""
import struct
import zlib

import numpy as np

MI_INT8, MI_UINT8, MI_DOUBLE, MI_MATRIX = 1, 2, 9, 14
MI_INT32, MI_UINT32, MI_CHAR, MI_CELL, MI_STRUCT = 5, 6, 16, 17, 2


def read_var(buf, pos):
    """读一个数据元素, 返回 (值, 新pos)。"""
    tag, = struct.unpack_from("<I", buf, pos)
    small = (tag >> 16) != 0
    if small:
        dtype, nbytes = tag & 0xFFFF, tag >> 16
        dstart, npos = pos + 4, pos + 8
    else:
        dtype, nbytes = tag & 0xFFFFFFFF, struct.unpack_from("<I", buf, pos + 4)[0]
        dstart, npos = pos + 8, pos + 8 + nbytes + (-nbytes % 8)
    data = buf[dstart:dstart + nbytes]
    if dtype == 15:  # miCOMPRESSED
        inner = zlib.decompress(data)
        # 解压流开头自带 miMATRIX 元素(tag+length 共 8 字节), 去掉后按矩阵体解析
        v = parse_matrix(inner[8:])
        return v, npos
    if dtype == MI_MATRIX:
        v = parse_matrix(data)
    elif dtype in (MI_DOUBLE,):
        v = np.frombuffer(data, "<f8").copy()
    elif dtype == MI_INT32:
        v = np.frombuffer(data, "<i4").copy()
    elif dtype == MI_UINT32:
        v = np.frombuffer(data, "<u4").copy()
    elif dtype == 3:    # miINT16
        v = np.frombuffer(data, "<i2").copy()
    elif dtype == 4:    # miUINT16
        v = np.frombuffer(data, "<u2").copy()
    elif dtype in (MI_INT8, MI_UINT8):
        v = data
    elif dtype == MI_CHAR:
        v = data.decode("utf-16-le", "ignore") if len(data) % 2 == 0 else data.decode("latin1")
    else:
        v = data
    return v, npos


def rd_el(buf, pos):
    tag, = struct.unpack_from("<I", buf, pos)
    small = (tag >> 16) != 0
    if small:
        dtype, nbytes = tag & 0xFFFF, tag >> 16
        return (dtype, buf[pos + 4:pos + 4 + nbytes], pos + 8)
    dtype, nbytes = tag, struct.unpack_from("<I", buf, pos + 4)[0]
    return (dtype, buf[pos + 8:pos + 8 + nbytes], pos + 8 + nbytes + (-nbytes % 8))


def parse_matrix(data):
    # 元素依次: flags(dims+flags), dims, name, 然后 data
    pos = 0
    dt, d, pos = rd_el(data, pos)          # array flags
    flags, = struct.unpack_from("<I", d, 0)
    cls = flags & 0xFF
    dt, dims_d, pos = rd_el(data, pos)      # dims
    dims = np.frombuffer(dims_d, "<i4")
    dt, name_d, pos = rd_el(data, pos)      # name
    name = name_d.decode("latin1")
    if cls == 2:  # struct (mxSTRUCT_CLASS = 2)
        dt, fnlen_d, pos = rd_el(data, pos)
        fnlen = int(np.frombuffer(fnlen_d, "<i4")[0])
        dt, fdata, pos = rd_el(data, pos)
        # 字段名: 单个 miINT8 元素内所有名字连续拼接, 每个占 fnlen 字节
        raw_names = fdata if dt in (1, 2) else None
        if raw_names is None:
            raise ValueError("unexpected field-name element dtype %d" % dt)
        n_names = len(raw_names) // fnlen
        fields = [raw_names[i * fnlen:(i + 1) * fnlen].rstrip(b"\x00 ").decode("latin1")
                  for i in range(n_names)]
        fdata = data[pos:]
        nelem = int(np.prod(dims)) or 1
        values = {f: [] for f in fields}
        fpos = 0
        for _ in range(nelem):
            for f in fields:
                v, fpos = read_var(fdata, fpos)
                values[f].append(v)
        return {f: (v[0] if len(v) == 1 else v) for f, v in values.items()}
    if cls == 4:   # char
        dt, cd, pos = rd_el(data, pos)
        if dt in (1, 2) or all(b < 128 for b in cd[:200]):
            return cd.decode("latin1")
        return cd.decode("utf-16-le", "ignore")
    if cls == 1:   # cell
        out = []
        p = pos
        while p < len(data):
            v, p = read_var(data, p)
            out.append(v)
        return out[0] if len(out) == 1 else out
    # numeric
    dt, nd, pos = rd_el(data, pos)
    if dt == MI_DOUBLE:
        a = np.frombuffer(nd, "<f8").copy()
    elif dt == MI_INT32:
        a = np.frombuffer(nd, "<i4").copy()
    elif dt == MI_UINT32:
        a = np.frombuffer(nd, "<u4").copy()
    elif dt == 3:    # miINT16
        a = np.frombuffer(nd, "<i2").copy()
    elif dt == 4:    # miUINT16
        a = np.frombuffer(nd, "<u2").copy()
    elif dt == MI_INT8:
        a = np.frombuffer(nd, "i1").copy()
    else:
        a = np.frombuffer(nd, "u1").copy()
    try:
        return a.reshape(dims, order="F")
    except ValueError:
        return a


def load_meta(path):
    """读 SUNRGBDMeta.mat -> dict(K=[3x3 数组]x10335, sequenceName=[...], ...)。"""
    buf = open(path, "rb").read()
    pos = 128  # 跳过文件头
    while pos < len(buf):
        v, pos = read_var(buf, pos)
        if isinstance(v, dict) and v:
            return v
    return {}
