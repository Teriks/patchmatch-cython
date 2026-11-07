# BSD 3-Clause License
#
# Copyright (c) 2025, Teriks
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c++
"""Xavier Philippeau pyramid inpainting, in Cython.

Numeric behavior is the Philippeau pyramid (bidirectional fields, texture
energy, split votes, border blend). ``patch_size`` is a radius.
"""

from libc.math cimport fabsf, nearbyint, sqrtf
from libc.stdlib cimport abs
from libcpp.random cimport mt19937

import operator
import secrets

import numpy as np
cimport numpy as np


cdef enum:
    DISTANCE_SCALE = 65535
    SSD_SCALE = 9 * 255 * 255
    MAX_RETRY = 20
    COARSE_PATCH = 5
    LEVEL_SIDES = 6
    MIN_FITTING = 7
    FEATURES = 12
cdef int KERNEL[6]
KERNEL[:] = [1, 5, 10, 10, 5, 1]

cdef mt19937 RNG
cdef double SIM[65536]
cdef bint SIM_OK = 0


cdef inline int clampi(int v, int lo, int hi) noexcept nogil:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


cdef inline double clampd(double v, double lo, double hi) noexcept nogil:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


cdef inline int clip_i(long long value, int limit) noexcept nogil:
    if value < 0:
        return 0
    if value > limit:
        return limit
    return <int>value


cdef inline unsigned char sat_u8(double value) noexcept nogil:
    value = nearbyint(value)
    if value < 0:
        return 0
    if value > 255:
        return 255
    return <unsigned char>value


cdef inline int random_int(int n) noexcept nogil:
    return <int>(RNG() % <unsigned int>n)


def _init_sim():
    cdef int i, j
    cdef double t
    cdef double base[11]
    base[0] = 1.0
    base[1] = 0.99
    base[2] = 0.96
    base[3] = 0.83
    base[4] = 0.38
    base[5] = 0.11
    base[6] = 0.02
    base[7] = 0.005
    base[8] = 0.0006
    base[9] = 0.0001
    base[10] = 0.0
    for i in range(65536):
        t = <double>i / 65536.0
        j = <int>(100.0 * t)
        if j >= 10:
            SIM[i] = 0.0
        else:
            SIM[i] = base[j] + (100.0 * t - j) * (base[j + 1] - base[j])


_init_sim()


cdef class Image:
    cdef object color_owner, mask_owner, global_owner, texture_owner, feat_owner, hole_owner
    cdef unsigned char[:, :, ::1] color
    cdef unsigned char[:, ::1] mask
    cdef unsigned char[:, ::1] global_mask
    cdef unsigned char[:, ::1] texture
    cdef unsigned char[:, :, ::1] features
    cdef int[:, ::1] hole_sums
    cdef int h, w
    cdef bint has_global, has_texture, features_ok, holes_ok

    cdef void bind_color(self, object color):
        self.color_owner = color
        self.color = color
        self.h = color.shape[0]
        self.w = color.shape[1]

    cdef bint is_masked(self, int y, int x) noexcept nogil:
        return self.mask[y, x] != 0

    cdef bint is_global(self, int y, int x) noexcept nogil:
        return self.has_global and self.global_mask[y, x] != 0

    cdef bint is_known(self, int y, int x) noexcept nogil:
        return (not self.is_masked(y, x)) and (not self.is_global(y, x))

    cdef bint is_hole(self, int y, int x) noexcept nogil:
        return self.is_masked(y, x) and (not self.is_global(y, x))

    cdef void clear_mask(self):
        self.mask[:] = 0
        self.holes_ok = False
        self.features_ok = False

    cdef void set_mask(self, int y, int x, bint value):
        self.mask[y, x] = 1 if value else 0
        self.holes_ok = False
        self.features_ok = False

    cdef void set_global(self, int y, int x, bint value):
        self.global_mask[y, x] = 1 if value else 0
        self.holes_ok = False
        self.features_ok = False

    cdef void set_texture(self, object texture):
        self.texture_owner = texture
        self.texture = texture
        self.has_texture = True
        self.features_ok = False

    cdef void ensure_holes(self):
        cdef int y, x, hole
        if self.holes_ok:
            return
        self.hole_owner = np.zeros((self.h + 1, self.w + 1), dtype=np.int32)
        self.hole_sums = self.hole_owner
        for y in range(self.h):
            for x in range(self.w):
                hole = 1 if self.is_hole(y, x) else 0
                self.hole_sums[y + 1, x + 1] = (
                    hole + self.hole_sums[y, x + 1] + self.hole_sums[y + 1, x] - self.hole_sums[y, x]
                )
        self.holes_ok = True

    cdef bint contains_mask(self, int y, int x, int patch) noexcept nogil:
        cdef int top, bottom, left, right
        top = clip_i(<long long>y - patch, self.h)
        bottom = clip_i(<long long>y + patch + 1, self.h)
        left = clip_i(<long long>x - patch, self.w)
        right = clip_i(<long long>x + patch + 1, self.w)
        return (
            self.hole_sums[bottom, right] - self.hole_sums[top, right]
            - self.hole_sums[bottom, left] + self.hole_sums[top, left]
        ) > 0

    cdef void compute_features(self):
        cdef int y, x, c, up, down, left, right
        cdef bint skip_x, skip_y
        cdef unsigned char[:, :, ::1] feat
        cdef unsigned char[:, :, ::1] color = self.color
        if self.features_ok:
            return
        self.feat_owner = np.zeros((self.h, self.w, FEATURES), dtype=np.uint8)
        self.features = self.feat_owner
        feat = self.features
        for y in range(self.h):
            up = y - 1 if y > 0 else 0
            down = y + 1 if y + 1 < self.h else self.h - 1
            for x in range(self.w):
                left = x - 1 if x > 0 else 0
                right = x + 1 if x + 1 < self.w else self.w - 1
                skip_y = self.is_global(down, x) or self.is_global(up, x)
                skip_x = self.is_global(y, right) or self.is_global(y, left)
                for c in range(3):
                    feat[y, x, c] = color[y, x, c]
                    if skip_x:
                        feat[y, x, 3 + c] = 128
                    else:
                        feat[y, x, 3 + c] = <unsigned char>(
                            <int>color[y, right, c] // 2 - <int>color[y, left, c] // 2 + 128
                        )
                    if skip_y:
                        feat[y, x, 6 + c] = 128
                    else:
                        feat[y, x, 6 + c] = <unsigned char>(
                            <int>color[down, x, c] // 2 - <int>color[up, x, c] // 2 + 128
                        )
                if self.has_texture:
                    feat[y, x, 9] = self.texture[y, x]
                feat[y, x, 10] = 1 if (self.is_masked(y, x) or self.is_global(y, x)) else 0
        self.features_ok = True

    cdef Image cloned(self):
        cdef Image c = Image()
        c.bind_color(np.array(self.color_owner, copy=True))
        c.mask_owner = np.array(self.mask_owner, copy=True)
        c.mask = c.mask_owner
        c.has_global = self.has_global
        if self.has_global:
            c.global_owner = np.array(self.global_owner, copy=True)
            c.global_mask = c.global_owner
        if self.has_texture:
            c.texture_owner = self.texture_owner
            c.texture = self.texture
            c.has_texture = True
        return c


cdef Image blank(int h, int w, bint with_global):
    cdef Image im = Image()
    im.bind_color(np.zeros((h, w, 3), dtype=np.uint8))
    im.mask_owner = np.zeros((h, w), dtype=np.uint8)
    im.mask = im.mask_owner
    if with_global:
        im.global_owner = np.zeros((h, w), dtype=np.uint8)
        im.global_mask = im.global_owner
        im.has_global = True
    return im


cdef Image from_inputs(unsigned char[:, :, ::1] image, unsigned char[:, ::1] mask, object global_mask):
    cdef Image im = Image()
    cdef int y, x
    im.bind_color(np.array(image, copy=True))
    im.mask_owner = np.array(mask, copy=True)
    im.mask = im.mask_owner
    if global_mask is not None:
        im.global_owner = np.array(global_mask, copy=True)
        im.global_mask = im.global_owner
        im.has_global = True
    for y in range(im.h):
        for x in range(im.w):
            if im.mask[y, x]:
                im.color[y, x, 0] = 0
                im.color[y, x, 1] = 0
                im.color[y, x, 2] = 0
    return im


cdef Image initial_image(unsigned char[:, :, ::1] image, unsigned char[:, ::1] mask, object global_mask):
    cdef Image im = Image()
    im.bind_color(np.array(image, copy=True))
    im.mask_owner = np.array(mask, copy=True)
    im.mask = im.mask_owner
    if global_mask is not None:
        im.global_owner = np.array(global_mask, copy=True)
        im.global_mask = im.global_owner
        im.has_global = True
    return im


cdef inline int pixel_cost(
    unsigned char *srow, unsigned char *trow, int sw, int tw, int xs, int xt, bint textured,
) noexcept nogil:
    cdef unsigned char *sp
    cdef unsigned char *tp
    cdef int d, ssd, gap
    if xs < 0 or xs >= sw or xt < 0 or xt >= tw:
        return SSD_SCALE
    sp = srow + xs * FEATURES
    tp = trow + xt * FEATURES
    if sp[10] or tp[10]:
        return SSD_SCALE
    d = <int>sp[0] - <int>tp[0]
    ssd = d * d
    d = <int>sp[1] - <int>tp[1]
    ssd += d * d
    d = <int>sp[2] - <int>tp[2]
    ssd += d * d
    d = <int>sp[3] - <int>tp[3]
    ssd += d * d
    d = <int>sp[4] - <int>tp[4]
    ssd += d * d
    d = <int>sp[5] - <int>tp[5]
    ssd += d * d
    d = <int>sp[6] - <int>tp[6]
    ssd += d * d
    d = <int>sp[7] - <int>tp[7]
    ssd += d * d
    d = <int>sp[8] - <int>tp[8]
    ssd += d * d
    if textured:
        gap = <int>sp[9] // 2 - <int>tp[9]
        if gap > 0:
            ssd += 2 * gap * gap
    return ssd


cdef inline long long row_sum(
    unsigned char *sf, unsigned char *tf, int sh, int sw, int th, int tw,
    int sy, int sx, int ty, int tx, int patch, bint textured,
) noexcept nogil:
    cdef int dx
    cdef long long total = 0
    cdef unsigned char *srow
    cdef unsigned char *trow
    if sy < 0 or sy >= sh or ty < 0 or ty >= th:
        return <long long>SSD_SCALE * (2 * patch + 1)
    srow = sf + sy * sw * FEATURES
    trow = tf + ty * tw * FEATURES
    for dx in range(-patch, patch + 1):
        total += pixel_cost(srow, trow, sw, tw, sx + dx, tx + dx, textured)
    return total


cdef inline long long col_sum(
    unsigned char *sf, unsigned char *tf, int sh, int sw, int th, int tw,
    int sy, int sx, int ty, int tx, int patch, bint textured,
) noexcept nogil:
    cdef int dy, ys, yt
    cdef long long total = 0
    for dy in range(-patch, patch + 1):
        ys = sy + dy
        yt = ty + dy
        if ys >= 0 and ys < sh and yt >= 0 and yt < th:
            total += pixel_cost(sf + ys * sw * FEATURES, tf + yt * tw * FEATURES, sw, tw, sx, tx, textured)
        else:
            total += SSD_SCALE
    return total


cdef long long patch_sum(Image source, Image target, int sy, int sx, int ty, int tx, int patch, int bound) noexcept nogil:
    cdef long long total = 0
    cdef int dy
    cdef double side = 2.0 * patch + 1.0
    cdef double limit = (bound + 1.0) * SSD_SCALE * side * side / DISTANCE_SCALE
    cdef unsigned char *sf = &source.features[0, 0, 0]
    cdef unsigned char *tf = &target.features[0, 0, 0]
    cdef bint textured = source.has_texture and target.has_texture
    cdef int sh = source.h
    cdef int sw = source.w
    cdef int th = target.h
    cdef int tw = target.w
    for dy in range(-patch, patch + 1):
        if <double>total > limit:
            return -1
        total += row_sum(sf, tf, sh, sw, th, tw, sy + dy, sx, ty + dy, tx, patch, textured)
    return total


cdef int scale_sum(long long total, int patch) noexcept nogil:
    cdef long long side = 2 * patch + 1
    cdef int res = <int>(DISTANCE_SCALE * (<double>total / SSD_SCALE) / <double>(side * side))
    if res < 0 or res > DISTANCE_SCALE:
        return DISTANCE_SCALE
    return res


cdef int regular_distance(Field field, int sy, int sx, int ty, int tx) noexcept nogil:
    cdef int my, mx, qy, qx
    cdef float di, dj
    cdef double score1, score2, score
    cdef long long psum
    if ty < 0 or ty >= field.target.h or tx < 0 or tx >= field.target.w:
        return DISTANCE_SCALE
    score1 = DISTANCE_SCALE
    if (not field.source.is_global(sy, sx)) and (not field.target.is_global(ty, tx)):
        my = <int>(<long long>sy * field.ij_h / field.source.h)
        mx = <int>(<long long>sx * field.ij_w / field.source.w)
        qy = <int>(<long long>ty * field.ij_h / field.target.h)
        qx = <int>(<long long>tx * field.ij_w / field.target.w)
        di = fabsf(field.ij[my, mx, 0] - field.ij[qy, qx, 0])
        if di > 0.5:
            di = 1.0 - di
        dj = fabsf(field.ij[my, mx, 1] - field.ij[qy, qx, 1])
        if dj > 0.5:
            dj = 1.0 - dj
        score1 = sqrtf(di * di + dj * dj) / 0.707
        if score1 < 0 or score1 > 1:
            score1 = 1
        score1 *= DISTANCE_SCALE
    psum = patch_sum(field.source, field.target, sy, sx, ty, tx, field.patch, DISTANCE_SCALE)
    score2 = DISTANCE_SCALE if psum < 0 else scale_sum(psum, field.patch)
    score = (score1 * field.guide_weight + score2) / (1.0 + field.guide_weight)
    if not (score > 0):
        return 0
    if score >= DISTANCE_SCALE:
        return DISTANCE_SCALE
    return <int>score


cdef class Field:
    cdef Image source, target
    cdef int patch
    cdef bint use_sums, regular
    cdef double guide_weight
    cdef float[:, :, ::1] ij
    cdef int ij_h, ij_w
    cdef object field_owner, sum_owner, ij_owner
    cdef int[:, :, ::1] links
    cdef long long[::1] sums

    cdef void _alloc(self):
        self.field_owner = np.zeros((self.source.h, self.source.w, 3), dtype=np.int32)
        self.links = self.field_owner
        if self.use_sums:
            self.sum_owner = np.full((self.source.h * self.source.w,), -1, dtype=np.int64)
            self.sums = self.sum_owner

    cdef int at(self, int y, int x, int c) noexcept nogil:
        return self.links[y, x, c]

    cdef void set_link(self, int y, int x, int ty, int tx, int distance, long long total) noexcept nogil:
        self.links[y, x, 0] = ty
        self.links[y, x, 1] = tx
        self.links[y, x, 2] = distance
        if self.use_sums:
            self.sums[y * self.source.w + x] = total

    cdef void measure_into(self, int y, int x, int ty, int tx, int bound, int *distance, long long *total) noexcept nogil:
        cdef long long psum
        if self.regular:
            distance[0] = regular_distance(self, y, x, ty, tx)
            total[0] = -1
            return
        psum = patch_sum(self.source, self.target, y, x, ty, tx, self.patch, bound)
        if psum < 0:
            distance[0] = DISTANCE_SCALE
            total[0] = -1
        else:
            distance[0] = scale_sum(psum, self.patch)
            total[0] = psum

    cdef void link_if_closer(self, int y, int x, int ty, int tx) noexcept nogil:
        cdef int distance
        cdef long long total
        cdef int current
        if ty == self.links[y, x, 0] and tx == self.links[y, x, 1]:
            return
        current = self.links[y, x, 2]
        self.measure_into(y, x, ty, tx, current, &distance, &total)
        if distance < current:
            self.set_link(y, x, ty, tx, distance, total)

    cdef void set_identity(self, int y, int x) noexcept nogil:
        self.set_link(y, x, y, x, 0, -1)

    cdef void update_distance(self, int y, int x) noexcept nogil:
        cdef int distance
        cdef long long total
        cdef int ty = self.links[y, x, 0]
        cdef int tx = self.links[y, x, 1]
        self.measure_into(y, x, ty, tx, DISTANCE_SCALE, &distance, &total)
        self.set_link(y, x, ty, tx, distance, total)

    cdef void prepare(self):
        self.source.ensure_holes()
        self.target.ensure_holes()
        self.source.compute_features()
        self.target.compute_features()

    cdef void randomize_link(self, int y, int x) noexcept nogil:
        cdef int t, ty = 0, tx = 0, distance
        cdef long long total
        distance = DISTANCE_SCALE
        total = -1
        for t in range(MAX_RETRY):
            ty = random_int(self.target.h)
            tx = random_int(self.target.w)
            if self.target.is_global(ty, tx):
                total = -1
                continue
            self.measure_into(y, x, ty, tx, DISTANCE_SCALE, &distance, &total)
            if distance < DISTANCE_SCALE:
                break
        self.set_link(y, x, ty, tx, distance, total)

    cdef void randomize_field(self, bint reset) noexcept nogil:
        cdef int y, x
        for y in range(self.source.h):
            for x in range(self.source.w):
                if self.source.is_global(y, x):
                    continue
                if (not reset) and self.links[y, x, 2] < DISTANCE_SCALE:
                    continue
                self.randomize_link(y, x)

    cdef void propagate(self, int y, int x, int dy, int dx) noexcept nogil:
        cdef int ny = y - dy
        cdef int nx = x - dx
        cdef int ty = self.links[ny, nx, 0] + dy
        cdef int tx = self.links[ny, nx, 1] + dx
        cdef long long neighbor, total
        cdef int leaving, distance
        cdef unsigned char *sf
        cdef unsigned char *tf
        cdef bint textured
        cdef int sh, sw, th, tw
        if not self.use_sums:
            self.link_if_closer(y, x, ty, tx)
            return
        neighbor = self.sums[ny * self.source.w + nx]
        if neighbor < 0:
            self.link_if_closer(y, x, ty, tx)
            return
        if ty == self.links[y, x, 0] and tx == self.links[y, x, 1]:
            return
        sf = &self.source.features[0, 0, 0]
        tf = &self.target.features[0, 0, 0]
        textured = self.source.has_texture and self.target.has_texture
        sh = self.source.h
        sw = self.source.w
        th = self.target.h
        tw = self.target.w
        leaving = self.patch + 1
        if dy != 0:
            total = neighbor
            total -= row_sum(sf, tf, sh, sw, th, tw,
                             y - dy * leaving, x - dx * leaving, ty - dy * leaving, tx - dx * leaving,
                             self.patch, textured)
            total += row_sum(sf, tf, sh, sw, th, tw,
                             y + dy * self.patch, x + dx * self.patch, ty + dy * self.patch, tx + dx * self.patch,
                             self.patch, textured)
        else:
            total = neighbor
            total -= col_sum(sf, tf, sh, sw, th, tw,
                             y - dy * leaving, x - dx * leaving, ty - dy * leaving, tx - dx * leaving,
                             self.patch, textured)
            total += col_sum(sf, tf, sh, sw, th, tw,
                             y + dy * self.patch, x + dx * self.patch, ty + dy * self.patch, tx + dx * self.patch,
                             self.patch, textured)
        distance = scale_sum(total, self.patch)
        if distance < self.links[y, x, 2]:
            self.set_link(y, x, ty, tx, distance, total)

    cdef void random_search(self, int y, int x) noexcept nogil:
        cdef int radius, yc, xc, top, left, height, width, yp, xp
        height = self.target.h
        width = self.target.w
        radius = (height if height < width else width) - 1
        radius //= 2
        while radius > 0:
            yc = clampi(self.links[y, x, 0], 0, height - 1)
            xc = clampi(self.links[y, x, 1], 0, width - 1)
            top = yc - radius if yc - radius > 0 else 0
            left = xc - radius if xc - radius > 0 else 0
            yp = top + random_int((yc + radius if yc + radius < height - 1 else height - 1) - top + 1)
            xp = left + random_int((xc + radius if xc + radius < width - 1 else width - 1) - left + 1)
            if not self.target.is_global(yp, xp):
                self.link_if_closer(y, x, yp, xp)
            radius //= 2

    cdef void init_from(self, Field other) noexcept nogil:
        cdef int y, x, ilow, jlow, ty, tx, distance
        cdef long long total
        cdef double fi = <double>self.source.h / other.source.h
        cdef double fj = <double>self.source.w / other.source.w
        cdef int patch = self.patch
        cdef bint linked
        for y in range(self.source.h):
            for x in range(self.source.w):
                if self.source.is_global(y, x):
                    continue
                if (not self.source.contains_mask(y, x, patch)) and (not self.target.contains_mask(y, x, patch)):
                    self.set_identity(y, x)
                    continue
                ilow = <int>(y / fi if y / fi < other.source.h - 1 else other.source.h - 1)
                jlow = <int>(x / fj if x / fj < other.source.w - 1 else other.source.w - 1)
                ty = clampi(<int>(other.links[ilow, jlow, 0] * fi + (y - ilow * fi)), 0, self.target.h - 1)
                tx = clampi(<int>(other.links[ilow, jlow, 1] * fj + (x - jlow * fj)), 0, self.target.w - 1)
                self.measure_into(y, x, ty, tx, DISTANCE_SCALE, &distance, &total)
                self.set_link(y, x, ty, tx, distance, total)
                linked = other.links[ilow, jlow, 0] == ilow and other.links[ilow, jlow, 1] == jlow
                if linked and distance > 0 and self.target.contains_mask(ty, tx, patch):
                    self.random_search(y, x)
        self.randomize_field(False)

    cdef void minimize_link(self, int y, int x, int direction) noexcept nogil:
        cdef int height = self.source.h
        cdef int width = self.source.w
        if self.source.is_global(y, x) or self.links[y, x, 2] <= 0:
            return
        if 0 <= y - direction < height and not self.source.is_global(y - direction, x):
            self.propagate(y, x, direction, 0)
        if 0 <= x - direction < width and not self.source.is_global(y, x - direction):
            self.propagate(y, x, 0, direction)
        self.random_search(y, x)

    cdef void minimize(self, int passes) noexcept nogil:
        cdef int y, x, n
        for n in range(passes):
            for y in range(self.source.h):
                for x in range(self.source.w):
                    self.minimize_link(y, x, 1)
            for y in range(self.source.h - 1, -1, -1):
                for x in range(self.source.w - 1, -1, -1):
                    self.minimize_link(y, x, -1)

    cdef void clear_sums(self) noexcept nogil:
        cdef int i
        if self.use_sums:
            for i in range(self.source.h * self.source.w):
                self.sums[i] = -1

    cdef Field copied(self):
        cdef Field c = Field()
        c.source = self.source
        c.target = self.target
        c.patch = self.patch
        c.use_sums = self.use_sums
        c.regular = self.regular
        c.guide_weight = self.guide_weight
        c.ij_h = self.ij_h
        c.ij_w = self.ij_w
        c.ij_owner = self.ij_owner
        if self.regular:
            c.ij = self.ij
        c.field_owner = np.array(self.field_owner, copy=True)
        c.links = c.field_owner
        if self.use_sums:
            c.sum_owner = np.array(self.sum_owner, copy=True)
            c.sums = c.sum_owner
        return c


cdef Field make_field(Image source, Image target, int patch, bint regular, double guide_weight,
                      object ij_owner, Field other):
    cdef Field f = Field()
    f.source = source
    f.target = target
    f.patch = patch
    f.regular = regular
    f.use_sums = not regular
    f.guide_weight = guide_weight
    if regular:
        f.ij_owner = ij_owner
        f.ij = ij_owner
        f.ij_h = ij_owner.shape[0]
        f.ij_w = ij_owner.shape[1]
    f._alloc()
    f.prepare()
    if other is None:
        with nogil:
            f.randomize_field(True)
    else:
        with nogil:
            f.init_from(other)
    return f


cdef Image downsample(Image image):
    cdef int half = 3
    cdef int y, x, dy, dx, yy, xx, k, weight
    cdef int r, g, b
    cdef bint globally
    cdef Image ret = blank(image.h // 2, image.w // 2, image.has_global)
    for y in range(0, image.h - 1, 2):
        for x in range(0, image.w - 1, 2):
            r = g = b = weight = 0
            globally = True
            for dy in range(-half + 1, half + 1):
                for dx in range(-half + 1, half + 1):
                    yy = y + dy
                    xx = x + dx
                    if yy < 0 or yy >= image.h or xx < 0 or xx >= image.w:
                        continue
                    if image.is_global(yy, xx):
                        continue
                    globally = False
                    if image.is_masked(yy, xx):
                        continue
                    k = KERNEL[half - 1 + dy] * KERNEL[half - 1 + dx]
                    r += image.color[yy, xx, 0] * k
                    g += image.color[yy, xx, 1] * k
                    b += image.color[yy, xx, 2] * k
                    weight += k
            if image.has_global:
                ret.set_global(y // 2, x // 2, globally)
            if weight == 0:
                ret.set_mask(y // 2, x // 2, True)
                continue
            ret.color[y // 2, x // 2, 0] = <unsigned char>(r // weight)
            ret.color[y // 2, x // 2, 1] = <unsigned char>(g // weight)
            ret.color[y // 2, x // 2, 2] = <unsigned char>(b // weight)
            ret.set_mask(y // 2, x // 2, False)
    return ret


cdef Image upsample(Image image, int new_h, int new_w, object fine_global):
    cdef Image ret = blank(new_h, new_w, image.has_global)
    cdef int y, x, yy, xx
    for y in range(new_h):
        for x in range(new_w):
            yy = <int>(<long long>y * image.h / new_h)
            xx = <int>(<long long>x * image.w / new_w)
            if image.is_global(yy, xx):
                ret.set_global(y, x, True)
                ret.set_mask(y, x, True)
                continue
            if image.has_global:
                ret.set_global(y, x, False)
            if image.is_masked(yy, xx):
                ret.set_mask(y, x, True)
                continue
            ret.color[y, x, 0] = image.color[yy, xx, 0]
            ret.color[y, x, 1] = image.color[yy, xx, 1]
            ret.color[y, x, 2] = image.color[yy, xx, 2]
            ret.set_mask(y, x, False)
    if fine_global is not None:
        ret.global_owner = fine_global
        ret.global_mask = fine_global
        ret.has_global = True
        ret.holes_ok = False
        ret.features_ok = False
    elif image.has_global and fine_global is None:
        pass
    return ret


cdef bint has_hole(Image image):
    cdef int y, x
    for y in range(image.h):
        for x in range(image.w):
            if image.is_hole(y, x):
                return True
    return False


cdef double grad_mag(Image image, int y, int x):
    cdef int up, down, left, right, c, magnitude
    up = y - 1 if y > 0 else 0
    down = y + 1 if y + 1 < image.h else image.h - 1
    left = x - 1 if x > 0 else 0
    right = x + 1 if x + 1 < image.w else image.w - 1
    if not (image.is_known(y, x) and image.is_known(up, x) and image.is_known(down, x)
            and image.is_known(y, left) and image.is_known(y, right)):
        return -1
    magnitude = 0
    for c in range(3):
        magnitude += abs(<int>image.color[down, x, c] - <int>image.color[up, x, c])
        magnitude += abs(<int>image.color[y, right, c] - <int>image.color[y, left, c])
    return magnitude / 2.0


cdef object texture_energy(Image image):
    cdef int radius = 2
    cdef int y, x, top, bottom, left, right
    cdef double mag
    cdef object sums_o = np.zeros((image.h + 1, image.w + 1, 2), dtype=np.float64)
    cdef object energy_o = np.full((image.h, image.w), -1.0, dtype=np.float64)
    cdef double[:, :, ::1] sums = sums_o
    cdef double[:, ::1] energy = energy_o
    cdef double s0, s1
    for y in range(image.h):
        for x in range(image.w):
            mag = grad_mag(image, y, x)
            if mag >= 0:
                s0 = mag
                s1 = 1.0
            else:
                s0 = 0.0
                s1 = 0.0
            sums[y + 1, x + 1, 0] = s0 + sums[y, x + 1, 0] + sums[y + 1, x, 0] - sums[y, x, 0]
            sums[y + 1, x + 1, 1] = s1 + sums[y, x + 1, 1] + sums[y + 1, x, 1] - sums[y, x, 1]
    for y in range(image.h):
        top = y - radius if y > radius else 0
        bottom = y + radius + 1 if y + radius + 1 < image.h else image.h
        for x in range(image.w):
            left = x - radius if x > radius else 0
            right = x + radius + 1 if x + radius + 1 < image.w else image.w
            s0 = sums[bottom, right, 0] - sums[top, right, 0] - sums[bottom, left, 0] + sums[top, left, 0]
            s1 = sums[bottom, right, 1] - sums[top, right, 1] - sums[bottom, left, 1] + sums[top, left, 1]
            if image.is_known(y, x) and s1 > 0:
                energy[y, x] = s0 / s1
    return energy_o


cdef double block_mean(double[:, ::1] values, int y, int x, int rows, int cols):
    cdef double total = 0
    cdef int count = 0
    cdef int yy, xx, y1, x1
    y1 = y + 2 if y + 2 < rows else rows
    x1 = x + 2 if x + 2 < cols else cols
    for yy in range(y, y1):
        for xx in range(x, x1):
            if values[yy, xx] >= 0:
                total += values[yy, xx]
                count += 1
    return total / count if count > 0 else -1


cdef double interpolate(double[:, ::1] coarse, int rows, int cols, int y, int x):
    cdef double cy, cx, fy, fx
    cdef int y0, x0, y1, x1
    cy = clampd((y + 0.5) / 2.0 - 0.5, 0.0, rows - 1.0)
    cx = clampd((x + 0.5) / 2.0 - 0.5, 0.0, cols - 1.0)
    y0 = <int>cy
    x0 = <int>cx
    y1 = y0 + 1 if y0 + 1 < rows else rows - 1
    x1 = x0 + 1 if x0 + 1 < cols else cols - 1
    fy = cy - y0
    fx = cx - x0
    return (1 - fy) * ((1 - fx) * coarse[y0, x0] + fx * coarse[y0, x1]) + fy * (
        (1 - fx) * coarse[y1, x0] + fx * coarse[y1, x1]
    )


cdef bint has_unknown(double[:, ::1] values, int rows, int cols):
    cdef int y, x
    for y in range(rows):
        for x in range(cols):
            if values[y, x] < 0:
                return True
    return False


cdef void pull_push(object values_o):
    cdef double[:, ::1] values = values_o
    cdef int rows = values_o.shape[0]
    cdef int cols = values_o.shape[1]
    cdef int y, x
    cdef object coarse_o
    cdef double[:, ::1] coarse
    if not has_unknown(values, rows, cols):
        return
    if rows * cols == 1:
        values[0, 0] = 0
        return
    coarse_o = np.zeros(((rows + 1) // 2, (cols + 1) // 2), dtype=np.float64)
    coarse = coarse_o
    for y in range(coarse.shape[0]):
        for x in range(coarse.shape[1]):
            coarse[y, x] = block_mean(values, 2 * y, 2 * x, rows, cols)
    pull_push(coarse_o)
    for y in range(rows):
        for x in range(cols):
            if values[y, x] < 0:
                values[y, x] = interpolate(coarse, coarse.shape[0], coarse.shape[1], y, x)


cdef object halve(object values_o, int height, int width):
    cdef double[:, ::1] values = values_o
    cdef int rows = values_o.shape[0]
    cdef int cols = values_o.shape[1]
    cdef object half_o = np.zeros((height, width), dtype=np.float64)
    cdef double[:, ::1] half = half_o
    cdef int y, x
    for y in range(height):
        for x in range(width):
            half[y, x] = block_mean(values, 2 * y, 2 * x, rows, cols)
    return half_o


cdef object to_bytes(object values_o):
    cdef double[:, ::1] values = values_o
    cdef object out_o = np.zeros(values_o.shape, dtype=np.uint8)
    cdef unsigned char[:, ::1] out = out_o
    cdef int y, x
    for y in range(values_o.shape[0]):
        for x in range(values_o.shape[1]):
            out[y, x] = sat_u8(values[y, x])
    return out_o


cdef void mean_window(Image image, unsigned char[:, :, ::1] color, int cy, int cx, int radius, bint known, double *out) noexcept nogil:
    cdef int y, x, y0, y1, x0, x1, count = 0, c
    cdef double s0 = 0, s1 = 0, s2 = 0
    y0 = cy - radius if cy > radius else 0
    x0 = cx - radius if cx > radius else 0
    y1 = cy + radius if cy + radius < image.h - 1 else image.h - 1
    x1 = cx + radius if cx + radius < image.w - 1 else image.w - 1
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if known:
                if not image.is_known(y, x):
                    continue
            else:
                if not image.is_hole(y, x):
                    continue
            s0 += color[y, x, 0]
            s1 += color[y, x, 1]
            s2 += color[y, x, 2]
            count += 1
    if count > 0:
        out[0] = s0 / count
        out[1] = s1 / count
        out[2] = s2 / count
    else:
        out[0] = out[1] = out[2] = 0


cdef void blend_borders(Image result, Image initial):
    cdef int width = 4
    cdef int y, x, d, dist, yy, xx
    cdef int[:, ::1] depth
    cdef double[:, :, ::1] steps
    cdef object depth_o = np.zeros((initial.h, initial.w), dtype=np.int32)
    cdef object steps_o = np.zeros((initial.h, initial.w, 3), dtype=np.float64)
    cdef double known_m[3]
    cdef double hole_m[3]
    cdef double acc0, acc1, acc2
    cdef int count, y0, y1, x0, x1
    depth = depth_o
    steps = steps_o
    for y in range(initial.h):
        for x in range(initial.w):
            if not initial.is_hole(y, x):
                continue
            d = 0
            y0 = y - width if y > width else 0
            x0 = x - width if x > width else 0
            y1 = y + width if y + width < initial.h - 1 else initial.h - 1
            x1 = x + width if x + width < initial.w - 1 else initial.w - 1
            for yy in range(y0, y1 + 1):
                for xx in range(x0, x1 + 1):
                    dist = abs(yy - y) if abs(yy - y) > abs(xx - x) else abs(xx - x)
                    if initial.is_known(yy, xx) and (d == 0 or dist < d):
                        d = dist
            depth[y, x] = d
    for y in range(initial.h):
        for x in range(initial.w):
            if depth[y, x] != 1:
                continue
            mean_window(initial, result.color, y, x, 1, True, known_m)
            mean_window(initial, result.color, y, x, 1, False, hole_m)
            steps[y, x, 0] = known_m[0] - hole_m[0]
            steps[y, x, 1] = known_m[1] - hole_m[1]
            steps[y, x, 2] = known_m[2] - hole_m[2]
    for y in range(result.h):
        for x in range(result.w):
            d = depth[y, x]
            if d == 0:
                continue
            acc0 = acc1 = acc2 = 0
            count = 0
            y0 = y - width if y > width else 0
            x0 = x - width if x > width else 0
            y1 = y + width if y + width < result.h - 1 else result.h - 1
            x1 = x + width if x + width < result.w - 1 else result.w - 1
            for yy in range(y0, y1 + 1):
                for xx in range(x0, x1 + 1):
                    if depth[yy, xx] == 1:
                        acc0 += steps[yy, xx, 0]
                        acc1 += steps[yy, xx, 1]
                        acc2 += steps[yy, xx, 2]
                        count += 1
            if count > 0:
                acc0 /= count
                acc1 /= count
                acc2 /= count
            acc0 *= <double>(width + 1 - d) / width
            acc1 *= <double>(width + 1 - d) / width
            acc2 *= <double>(width + 1 - d) / width
            result.color[y, x, 0] = sat_u8(result.color[y, x, 0] + acc0)
            result.color[y, x, 1] = sat_u8(result.color[y, x, 1] + acc1)
            result.color[y, x, 2] = sat_u8(result.color[y, x, 2] + acc2)


cdef class State:
    cdef list pyr
    cdef int npyr
    cdef int patch0
    cdef int fitting[64]
    cdef bint regular
    cdef double guide_weight
    cdef object ij_owner
    cdef Field s2t, t2s
    cdef bint has_s2t
    cdef int level_patch
    cdef int coarse_patch
    cdef int coarsest_hole


cdef void attach(Field f, Image source, Image target, bint source_side):
    if source_side:
        f.source = source
    else:
        f.target = target
    f.clear_sums()


cdef void update_links(State st, Image source, bint image_changed):
    cdef int y, x, patch
    if image_changed:
        source.compute_features()
        if st.has_s2t:
            st.s2t.source.compute_features()
            st.s2t.target.compute_features()
        st.t2s.source.compute_features()
        st.t2s.target.compute_features()
    if st.has_s2t:
        st.s2t.prepare()
    st.t2s.prepare()
    source.ensure_holes()
    patch = st.level_patch
    for y in range(source.h):
        for x in range(source.w):
            if not source.contains_mask(y, x, patch):
                if st.has_s2t:
                    st.s2t.set_identity(y, x)
                st.t2s.set_identity(y, x)
            elif image_changed and not source.is_global(y, x):
                if st.has_s2t:
                    st.s2t.update_distance(y, x)
                st.t2s.update_distance(y, x)


cdef object low_pass(Image image):
    cdef object out_o = np.zeros((image.h, image.w, 3), dtype=np.uint8)
    cdef unsigned char[:, :, ::1] out = out_o
    cdef int y, x
    cdef double mean[3]
    for y in range(image.h):
        for x in range(image.w):
            mean_window(image, image.color, y, x, 1, True, mean)
            out[y, x, 0] = sat_u8(mean[0])
            out[y, x, 1] = sat_u8(mean[1])
            out[y, x, 2] = sat_u8(mean[2])
    return out_o


cdef bint rect_has_mask(long long[:, ::1] sums, int height, int width,
                        long long y0, long long x0, long long y1, long long x1) noexcept nogil:
    cdef int top = clip_i(y0, height)
    cdef int bottom = clip_i(y1 + 1, height)
    cdef int left = clip_i(x0, width)
    cdef int right = clip_i(x1 + 1, width)
    return sums[bottom, right] - sums[top, right] - sums[bottom, left] + sums[top, left] > 0


cdef inline bint masked_at(unsigned char *mask, unsigned char *gmask, int w, int y, int x) noexcept nogil:
    cdef int i = y * w + x
    if mask[i]:
        return True
    return gmask != NULL and gmask[i] != 0


cdef void color_shift(Field field, int i, int j, int yp, int xp, int patch, bint source2target, double *shift) noexcept nogil:
    cdef int di, dj, ys, xs, yt, xt, count = 0, si, ti
    cdef double s0 = 0, s1 = 0, s2 = 0
    cdef double sign
    cdef int sh = field.source.h
    cdef int sw = field.source.w
    cdef int th = field.target.h
    cdef int tw = field.target.w
    cdef unsigned char *sc = &field.source.color[0, 0, 0]
    cdef unsigned char *tc = &field.target.color[0, 0, 0]
    cdef unsigned char *sm = &field.source.mask[0, 0]
    cdef unsigned char *tm = &field.target.mask[0, 0]
    cdef unsigned char *sg = NULL
    cdef unsigned char *tg = NULL
    if field.source.has_global:
        sg = &field.source.global_mask[0, 0]
    if field.target.has_global:
        tg = &field.target.global_mask[0, 0]
    for di in range(-patch, patch + 1):
        for dj in range(-patch, patch + 1):
            ys = i + di
            xs = j + dj
            yt = yp + di
            xt = xp + dj
            if ys < 0 or ys >= sh or xs < 0 or xs >= sw or yt < 0 or yt >= th or xt < 0 or xt >= tw:
                continue
            if masked_at(sm, sg, sw, ys, xs) or masked_at(tm, tg, tw, yt, xt):
                continue
            si = (ys * sw + xs) * 3
            ti = (yt * tw + xt) * 3
            s0 += sc[si] - tc[ti]
            s1 += sc[si + 1] - tc[ti + 1]
            s2 += sc[si + 2] - tc[ti + 2]
            count += 1
    if count == 0:
        shift[0] = shift[1] = shift[2] = 0
        return
    sign = (-1.0 if source2target else 1.0) / count
    shift[0] = s0 * sign
    shift[1] = s1 * sign
    shift[2] = s2 * sign


cdef inline void vote_copy(
    unsigned char *color, unsigned char *mask, unsigned char *gmask, unsigned char *texture, unsigned char *low,
    int sh, int sw, double *vote, int vh, int vw, int channels,
    int ys, int xs, int yt, int xt, double weight, double best_weight,
    double s0, double s1, double s2, bint best_only, bint split,
) noexcept nogil:
    cdef int pix, base
    cdef double *vp
    cdef unsigned char *cp
    cdef unsigned char *lp
    if ys < 0 or xs < 0 or yt < 0 or xt < 0 or ys >= sh or xs >= sw or yt >= vh or xt >= vw:
        return
    vp = vote + (yt * vw + xt) * channels
    pix = ys * sw + xs
    if best_only:
        if mask[pix]:
            return
        if best_weight <= vp[3]:
            return
        cp = color + pix * 3
        vp[0] = (cp[0] + s0) * best_weight
        vp[1] = (cp[1] + s1) * best_weight
        vp[2] = (cp[2] + s2) * best_weight
        vp[3] = best_weight
        return
    if split:
        if masked_at(mask, gmask, sw, ys, xs):
            return
        cp = color + pix * 3
        lp = low + pix * 3
        vp[0] += (lp[0] + s0) * weight
        vp[1] += (lp[1] + s1) * weight
        vp[2] += (lp[2] + s2) * weight
        vp[3] += weight
        vp[8] += texture[pix] * weight
        if best_weight <= vp[7]:
            return
        vp[4] = cp[0] - lp[0]
        vp[5] = cp[1] - lp[1]
        vp[6] = cp[2] - lp[2]
        vp[7] = best_weight
        return
    if masked_at(mask, gmask, sw, ys, xs):
        return
    base = pix * 3
    vp[0] += color[base] * weight
    vp[1] += color[base + 1] * weight
    vp[2] += color[base + 2] * weight
    vp[3] += weight


cdef void cast_field(Field field, bint source2target, double[:, :, ::1] vote, Image source,
                     bint upscaled, bint best_only, bint keep_known, int vote_patch, object low_owner):
    cdef int i, j, di, dj, ys, xs, yt, xt, yp, xp, uy, ux
    cdef int sy, sx
    cdef double weight, best_weight, offset2
    cdef double shift[3]
    cdef long long scale
    cdef int channels = vote.shape[2]
    cdef bint split = channels == 9
    cdef unsigned char[:, :, ::1] low
    cdef long long[:, ::1] holes
    cdef object hole_o
    cdef int hy, hx
    cdef int fsh = field.source.h
    cdef int fsw = field.source.w
    cdef int fth = field.target.h
    cdef int ftw = field.target.w
    cdef int sh = source.h
    cdef int sw = source.w
    cdef int vh = vote.shape[0]
    cdef int vw = vote.shape[1]
    cdef unsigned char *color = &source.color[0, 0, 0]
    cdef unsigned char *mask = &source.mask[0, 0]
    cdef unsigned char *gmask = NULL
    cdef unsigned char *texture = NULL
    cdef unsigned char *lowp = NULL
    cdef unsigned char *fs_g = NULL
    cdef unsigned char *ft_g = NULL
    cdef double *vote_p = &vote[0, 0, 0]
    cdef int *links = &field.links[0, 0, 0]
    cdef int *lk
    if source.has_global:
        gmask = &source.global_mask[0, 0]
    if source.has_texture:
        texture = &source.texture[0, 0]
    if field.source.has_global:
        fs_g = &field.source.global_mask[0, 0]
    if field.target.has_global:
        ft_g = &field.target.global_mask[0, 0]
    if not split:
        low_owner = np.zeros((1, 1, 3), dtype=np.uint8)
    low = low_owner
    if split:
        lowp = &low[0, 0, 0]
    if keep_known:
        hole_o = np.zeros((source.h + 1, source.w + 1), dtype=np.int64)
        holes = hole_o
        for hy in range(source.h):
            for hx in range(source.w):
                holes[hy + 1, hx + 1] = (1 if source.is_masked(hy, hx) else 0) + holes[hy, hx + 1] + holes[hy + 1, hx] - holes[hy, hx]
    for i in range(fsh):
        for j in range(fsw):
            if fs_g != NULL and fs_g[i * fsw + j]:
                continue
            lk = links + (i * fsw + j) * 3
            yp = lk[0]
            xp = lk[1]
            if keep_known:
                scale = 2 if upscaled else 1
                sy = yp if source2target else i
                sx = xp if source2target else j
                if not rect_has_mask(
                    holes, sh, sw,
                    scale * (<long long>sy - vote_patch),
                    scale * (<long long>sx - vote_patch),
                    scale * (<long long>sy + vote_patch + 1) - 1,
                    scale * (<long long>sx + vote_patch + 1) - 1,
                ):
                    continue
            if keep_known:
                color_shift(field, i, j, yp, xp, vote_patch, source2target, shift)
            else:
                shift[0] = shift[1] = shift[2] = 0
            weight = SIM[lk[2]]
            for di in range(-vote_patch, vote_patch + 1):
                for dj in range(-vote_patch, vote_patch + 1):
                    ys = i + di
                    xs = j + dj
                    yt = yp + di
                    xt = xp + dj
                    if ys < 0 or ys >= fsh or xs < 0 or xs >= fsw or (fs_g != NULL and fs_g[ys * fsw + xs]):
                        continue
                    if yt < 0 or yt >= fth or xt < 0 or xt >= ftw or (ft_g != NULL and ft_g[yt * ftw + xt]):
                        continue
                    offset2 = <double>di * di + <double>dj * dj
                    best_weight = weight * (1.0 - 1e-5 * offset2)
                    if not source2target:
                        ys, yt = yt, ys
                        xs, xt = xt, xs
                    if not upscaled:
                        vote_copy(
                            color, mask, gmask, texture, lowp, sh, sw, vote_p, vh, vw, channels,
                            ys, xs, yt, xt, weight, best_weight, shift[0], shift[1], shift[2], best_only, split,
                        )
                    else:
                        for uy in range(2):
                            for ux in range(2):
                                vote_copy(
                                    color, mask, gmask, texture, lowp, sh, sw, vote_p, vh, vw, channels,
                                    2 * ys + uy, 2 * xs + ux, 2 * yt + uy, 2 * xt + ux,
                                    weight, best_weight, shift[0], shift[1], shift[2], best_only, split,
                                )


cdef void maximize(Image target, double[:, :, ::1] vote, Image source, bint keep_known):
    cdef int y, x
    cdef bint split = vote.shape[2] == 9
    cdef double r, g, b
    for y in range(target.h):
        for x in range(target.w):
            if target.is_global(y, x):
                continue
            if keep_known and not source.is_masked(y, x):
                target.color[y, x, 0] = source.color[y, x, 0]
                target.color[y, x, 1] = source.color[y, x, 1]
                target.color[y, x, 2] = source.color[y, x, 2]
                continue
            if vote[y, x, 3] > 0:
                r = vote[y, x, 0] / vote[y, x, 3] + (vote[y, x, 4] if split else 0)
                g = vote[y, x, 1] / vote[y, x, 3] + (vote[y, x, 5] if split else 0)
                b = vote[y, x, 2] / vote[y, x, 3] + (vote[y, x, 6] if split else 0)
                target.color[y, x, 0] = sat_u8(r)
                target.color[y, x, 1] = sat_u8(g)
                target.color[y, x, 2] = sat_u8(b)


cdef object voted_texture(Image source, double[:, :, ::1] vote):
    cdef object tex = np.array(source.texture_owner, copy=True)
    cdef unsigned char[:, ::1] out = tex
    cdef int y, x
    for y in range(source.h):
        for x in range(source.w):
            if source.is_hole(y, x) and vote[y, x, 3] > 0:
                out[y, x] = sat_u8(2.0 * vote[y, x, 8] / vote[y, x, 3])
    return tex


cdef int nnf_iters(int level) noexcept nogil:
    cdef int n = 1 + level
    return 7 if n > 7 else n


cdef Image expectation(State st, Image source, Image target, int level):
    cdef int em_iters = 1 + 2 * level
    cdef int passes = nnf_iters(level)
    cdef int em, vote_patch, channels
    cdef bint upscaled, best_only, keep_known, split
    cdef Image new_source, new_target
    cdef object vote_o, low_o, fine_global
    cdef double[:, :, ::1] vote
    vote_patch = st.patch0 if st.patch0 < st.level_patch else st.level_patch
    for em in range(em_iters):
        if em != 0:
            if st.has_s2t:
                st.s2t.target = new_target
                st.s2t.clear_sums()
            st.t2s.source = new_target
            st.t2s.clear_sums()
            target = new_target
        update_links(st, source, em != 0)
        if st.has_s2t:
            with nogil:
                st.s2t.minimize(passes)
        with nogil:
            st.t2s.minimize(passes)
        upscaled = level >= 1 and em == em_iters - 1
        if upscaled:
            new_source = <Image>st.pyr[level - 1]
            fine_global = new_source.global_owner if new_source.has_global else None
            new_target = upsample(target, new_source.h, new_source.w, fine_global)
        else:
            new_source = <Image>st.pyr[level]
            new_target = target.cloned()
        best_only = level == 0 and em == em_iters - 1
        keep_known = (level - 1 if upscaled else level) <= 1
        split = keep_known and not best_only
        channels = 9 if split else 4
        vote_o = np.zeros((new_target.h, new_target.w, channels), dtype=np.float64)
        vote = vote_o
        low_o = low_pass(new_source) if split else None
        if st.has_s2t:
            cast_field(st.s2t, True, vote, new_source, upscaled, best_only, keep_known, vote_patch, low_o)
        cast_field(st.t2s, False, vote, new_source, upscaled, best_only, keep_known, vote_patch, low_o)
        maximize(new_target, vote, new_source, keep_known)
        if upscaled and level == 1:
            new_target.set_texture(voted_texture(new_source, vote))
        elif new_source.has_texture:
            new_target.set_texture(new_source.texture_owner)
    return new_target


cdef Image run_levels(State st, int level_from, int level_to, Image target, int metric_patch):
    cdef int level, patch
    cdef bint coarsest
    cdef Image source
    for level in range(level_from, level_to - 1, -1):
        source = <Image>st.pyr[level]
        patch = st.fitting[level] if st.fitting[level] else metric_patch
        st.level_patch = patch
        coarsest = level == st.npyr - 1
        if coarsest:
            target = source.cloned()
            target.clear_mask()
        if level < 2:
            st.has_s2t = False
            st.s2t = None
        else:
            st.s2t = make_field(
                source, target, patch, st.regular, st.guide_weight, st.ij_owner,
                None if coarsest else st.s2t,
            )
            st.has_s2t = True
        st.t2s = make_field(
            target, source, patch, st.regular, st.guide_weight, st.ij_owner,
            None if coarsest else st.t2s,
        )
        target = expectation(st, source, target, level)
    return target


cdef double hole_energy(State st, int level, Image target):
    cdef Image source = <Image>st.pyr[level]
    cdef Field field = make_field(
        target, source, st.patch0, st.regular, st.guide_weight, st.ij_owner, st.t2s,
    )
    cdef int y, x, count = 0
    cdef double total = 0
    with nogil:
        field.minimize(nnf_iters(level))
    for y in range(source.h):
        for x in range(source.w):
            if source.contains_mask(y, x, st.patch0) and not source.is_global(y, x):
                total += field.links[y, x, 2]
                count += 1
    return total / count if count else 0


cdef Image fill_coarse(State st, Image target):
    cdef int level = st.coarsest_hole
    cdef Field saved_s = st.s2t.copied()
    cdef Field saved_t = st.t2s.copied()
    cdef Image plain = run_levels(st, level, level - 1, target, st.patch0)
    cdef double plain_energy = hole_energy(st, level - 2, plain)
    cdef Field plain_s = st.s2t.copied() if st.has_s2t else None
    cdef Field plain_t = st.t2s.copied()
    cdef bint plain_has = st.has_s2t
    cdef Image coarse
    cdef double coarse_energy
    st.s2t = saved_s
    st.t2s = saved_t
    st.has_s2t = True
    coarse = run_levels(st, level, level - 1, target, st.coarse_patch)
    coarse_energy = hole_energy(st, level - 2, coarse)
    st.level_patch = st.patch0
    if coarse_energy < plain_energy:
        return coarse
    st.s2t = plain_s
    st.t2s = plain_t
    st.has_s2t = plain_has
    return plain


cdef Image inpaint_image(Image initial, Image working, int patch, object ijmap, double guide_weight):
    cdef State st = State()
    cdef Image source, target, level
    cdef object texture
    cdef int side, fitting, level_i
    st.pyr = [working]
    source = working
    while source.h > patch and source.w > patch:
        source = downsample(source)
        st.pyr.append(source)
    st.npyr = len(st.pyr)
    texture = texture_energy(<Image>st.pyr[0])
    pull_push(texture)
    for level_i in range(st.npyr):
        level = <Image>st.pyr[level_i]
        if level_i:
            texture = halve(texture, level.h, level.w)
        level.set_texture(to_bytes(texture))
        if has_hole(level):
            st.coarsest_hole = level_i
        side = level.h if level.h < level.w else level.w
        fitting = MIN_FITTING if MIN_FITTING > (side // LEVEL_SIDES - 1) // 2 else (side // LEVEL_SIDES - 1) // 2
        st.fitting[level_i] = fitting if fitting < patch else 0
    st.patch0 = patch
    st.regular = ijmap is not None
    st.guide_weight = guide_weight
    st.ij_owner = ijmap
    st.coarse_patch = COARSE_PATCH if patch < COARSE_PATCH else 0
    level_i = st.npyr - 1
    target = None
    if st.coarse_patch and st.coarsest_hole >= 2:
        target = run_levels(st, level_i, st.coarsest_hole + 1, target, patch)
        target = fill_coarse(st, target)
        level_i = st.coarsest_hole - 2
    target = run_levels(st, level_i, 0, target, patch)
    return target


def inpaint_buffer(
    np.ndarray[np.uint8_t, ndim=3] image,
    np.ndarray[np.uint8_t, ndim=2] mask,
    object global_mask,
    object ijmap,
    int patch_size,
    double guide_weight,
    unsigned int seed,
):
    cdef Image initial, working, filled
    cdef int y, x
    cdef np.ndarray[np.uint8_t, ndim=3] out
    RNG.seed(seed)
    initial = initial_image(image, mask, global_mask)
    working = from_inputs(image, mask, global_mask)
    if initial.has_texture:
        pass
    filled = inpaint_image(initial, working, patch_size, ijmap, guide_weight)
    for y in range(initial.h):
        for x in range(initial.w):
            if initial.is_masked(y, x) and not initial.is_global(y, x):
                continue
            filled.color[y, x, 0] = initial.color[y, x, 0]
            filled.color[y, x, 1] = initial.color[y, x, 1]
            filled.color[y, x, 2] = initial.color[y, x, 2]
    blend_borders(filled, initial)
    out = np.array(filled.color_owner, copy=True)
    return out


_MAX_PATCH_SIZE = (2**31 - 2) // 2
_MAX_GUIDE_WEIGHT = float(np.finfo(np.float32).max)


def inpaint_philippeau(
    image,
    mask=None,
    *,
    global_mask=None,
    patch_size=15,
    seed=None,
    ijmap=None,
    guide_weight=0.25,
):
    """Fill holes with the Xavier Philippeau algorithm.

    ``image`` is an HxWx3 uint8 array. ``mask`` is HxW (or HxWx1) uint8 or
    bool; non-zero marks a hole. If ``mask`` is omitted, pure white pixels are
    the holes. ``global_mask`` marks pixels that are neither filled nor used as
    a source. ``ijmap`` is an HxWx3 float32 regularity map (only the first two
    channels are read) and ``guide_weight`` is its weight.

    Only hole pixels change. A given ``seed`` is deterministic (taken modulo
    ``2**32``). If ``seed`` is omitted, a new 32-bit seed is drawn for the run.
    """
    patch_size = _check_int("patch_size", patch_size, 1, _MAX_PATCH_SIZE)
    if seed is None:
        seed = secrets.randbits(32)
    seed = operator.index(seed) % 2**32
    image_u8 = _canon_image(image)
    mask_u8 = _default_mask(image_u8) if mask is None else _canon_mask(mask, "mask")
    global_u8 = None if global_mask is None else _canon_mask(global_mask, "global_mask")
    _check_same_size("mask", mask_u8, image_u8)
    if global_u8 is not None:
        _check_same_size("global_mask", global_u8, image_u8)

    ij = None
    if ijmap is not None:
        ij = np.asarray(ijmap)
        if not (isinstance(ij, np.ndarray) and ij.ndim == 3 and ij.shape[2] == 3 and ij.dtype == np.float32 and ij.size > 0):
            raise ValueError("ijmap must be a non-empty HxWx3 float32 array")
        if not np.isfinite(ij).all():
            raise ValueError("ijmap must only contain finite values")
        ij = np.ascontiguousarray(ij)
        guide_weight = float(guide_weight)
        if not 0 <= guide_weight <= _MAX_GUIDE_WEIGHT:
            raise ValueError(
                f"guide_weight must be between 0 and {_MAX_GUIDE_WEIGHT:g}, got {guide_weight}"
            )
    else:
        guide_weight = 0.0

    holes = mask_u8 != 0
    if global_u8 is not None:
        holes &= global_u8 == 0
    if image_u8.size == 0 or not holes.any():
        return image_u8.copy()

    return inpaint_buffer(
        image_u8,
        mask_u8,
        global_u8,
        ij,
        patch_size,
        float(guide_weight),
        seed,
    )


def _check_int(name, value, minimum, maximum):
    try:
        value = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}") from None
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}, got {value}")
    return value


def _canon_image(image):
    image = np.asarray(image)
    if not (image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8):
        raise ValueError(
            "image must be an HxWx3 uint8 array, "
            f"got shape {image.shape} and dtype {image.dtype}"
        )
    return np.ascontiguousarray(image)


def _canon_mask(mask, name):
    mask = np.asarray(mask)
    if mask.dtype == np.bool_:
        mask = mask.astype(np.uint8)
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]
    if not (mask.ndim == 2 and mask.dtype == np.uint8):
        raise ValueError(
            f"{name} must be an HxW or HxWx1 uint8 array, "
            f"got shape {mask.shape} and dtype {mask.dtype}"
        )
    return np.ascontiguousarray(mask)


def _default_mask(image):
    return (image == 255).all(axis=2).astype(np.uint8)


def _check_same_size(name, mask, image):
    if mask.shape[:2] != image.shape[:2]:
        raise ValueError(
            f"{name} must have the same height and width as the image, "
            f"got {mask.shape[:2]} for an image of {image.shape[:2]}"
        )
