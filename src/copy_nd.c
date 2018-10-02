#include <aml.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

static inline void aml_copy_2d_helper(void *dst, const size_t *cumul_dst_pitch, const void *src, const size_t *cumul_src_pitch, const size_t *elem_number, size_t elem_size)
{
        for(int i = 0; i < elem_number[1]; i++) {
                memcpy(dst, src, elem_number[0]*elem_size);
                dst += cumul_dst_pitch[0];
                src += cumul_src_pitch[0];
        }
}

static inline void aml_copy_nd_helper(size_t d, void *dst, const size_t *cumul_dst_pitch, const void *src, const size_t *cumul_src_pitch, const size_t *elem_number, const size_t elem_size) {
        if(d == 2)
                aml_copy_2d_helper(dst, cumul_dst_pitch, src, cumul_src_pitch, elem_number, elem_size);
        else {
                for(int i = 0; i < elem_number[d-1]; i++) {
                        aml_copy_nd_helper(d-1, dst, cumul_dst_pitch, src, cumul_src_pitch, elem_number, elem_size);
                        dst += cumul_dst_pitch[d-2];
                        src += cumul_src_pitch[d-2];
                }
        }
}

int aml_copy_nd(size_t d, void *dst, const size_t *dst_pitch, const void *src, const size_t *src_pitch, const size_t *elem_number, const size_t elem_size)
{
        assert(d > 0);
        if(d == 1)
                memcpy(dst, src, elem_number[0]*elem_size);
        else {
                size_t * cumul_dst_pitch = (size_t *)alloca((d-1)*sizeof(size_t));
                size_t * cumul_src_pitch = (size_t *)alloca((d-1)*sizeof(size_t));
                assert(dst_pitch[0] >= elem_number[0]);
                assert(src_pitch[0] >= elem_number[0]);
                cumul_dst_pitch[0] = elem_size * dst_pitch[0];
                cumul_src_pitch[0] = elem_size * src_pitch[0];
                for(int i = 1; i < d - 1; i++) {
                        assert(dst_pitch[i] >= elem_number[i]);
                        assert(src_pitch[i] >= elem_number[i]);
                        cumul_dst_pitch[i] = dst_pitch[i] * cumul_dst_pitch[i-1];
                        cumul_src_pitch[i] = src_pitch[i] * cumul_src_pitch[i-1];
                }
                aml_copy_nd_helper(d, dst, cumul_dst_pitch, src, cumul_src_pitch, elem_number, elem_size);
        }
        return 0;
}


static inline void aml_copy_t2d_helper(void *dst, const size_t *cumul_dst_pitch, const void *src, const size_t *cumul_src_pitch, const size_t *elem_number, const size_t elem_size)
{
        for(int j = 0; j < elem_number[1]; j++)
                for(int i = 0; i < elem_number[0]; i++)
                        memcpy(dst + j * elem_size + i * cumul_dst_pitch[0],
                               src + i * elem_size + j * cumul_src_pitch[0],
                               elem_size);
}

// d[j + i * dp[0]]
// d[j + k * dp[0] + i * dp[0] * dp[1]]
// d[j + k * dp[0] + l * dp[0] * dp[1] + i * dp[0] * dp[1] * dp[2]]
// s[i + j * sp[0]]
// s[i + j * sp[0] + k * sp[0] * sp[1]]
// s[i + j * sp[0] + k * sp[0] * sp[1] + l * sp[0] * sp[1] * sp[2]]
static void aml_copy_tnd_helper(size_t d, void *dst, size_t *cumul_dst_pitch, const void *src, const size_t *cumul_src_pitch, const size_t *elem_number, const size_t elem_size)
{
        if(d == 2)
                aml_copy_t2d_helper(dst, cumul_dst_pitch, src, cumul_src_pitch, elem_number, elem_size);
        else {
                size_t * new_cumul_dst_pitch = (size_t *)alloca((d-1)*sizeof(size_t));
                // process dimension d-1
                for(int i = 0; i < d-3; i++)
                        new_cumul_dst_pitch[i] = cumul_dst_pitch[i];
                new_cumul_dst_pitch[d-3] = cumul_dst_pitch[d-2];
                for(int i = 0; i < elem_number[d-1]; i++) {
                        aml_copy_tnd_helper(d-1, dst, new_cumul_dst_pitch, src, cumul_src_pitch, elem_number, elem_size);
                        dst += cumul_dst_pitch[d-3];
                        src += cumul_src_pitch[d-2];
                }
        }
}

int aml_copy_tnd(size_t d, void *dst, const size_t *dst_pitch, const void *src, const size_t *src_pitch, const size_t *elem_number, const size_t elem_size)
{
        assert(d > 0);
        if(d == 1)
                memcpy(dst, src, elem_number[0]*elem_size);
        else {
                size_t * cumul_dst_pitch = (size_t *)alloca((d-1)*sizeof(size_t));
                size_t * cumul_src_pitch = (size_t *)alloca((d-1)*sizeof(size_t));
                assert(dst_pitch[0] >= elem_number[1]);
                assert(src_pitch[0] >= elem_number[0]);
                cumul_dst_pitch[0] = elem_size * dst_pitch[0];
                cumul_src_pitch[0] = elem_size * src_pitch[0];
                for(int i = 1; i < d - 1; i++) {
                        assert(dst_pitch[i] >= elem_number[i+1]);
                        assert(src_pitch[i] >= elem_number[i]);
                        cumul_dst_pitch[i] = dst_pitch[i] * cumul_dst_pitch[i-1];
                        cumul_src_pitch[i] = src_pitch[i] * cumul_src_pitch[i-1];
                }
                aml_copy_tnd_helper(d, dst, cumul_dst_pitch, src, cumul_src_pitch, elem_number, elem_size);
        }
        return 0;
}
// d[l + i * dp[0]]
// d[l + i * dp[0] + k * dp[0] * dp[1]]
// d[l + i * dp[0] + j * dp[0] * dp[1] + k * dp[0] * dp[1] * dp[2]]
// s[i + l * sp[0]]
// s[i + k * sp[0] + l * sp[0] * sp[1]]
// s[i + j * sp[0] + k * sp[0] * sp[1] + l * sp[0] * sp[1] * sp[2]]
static void aml_copy_rtnd_helper(size_t d, void *dst, const size_t *cumul_dst_pitch, const void *src, const size_t *cumul_src_pitch, const size_t *elem_number, const size_t elem_size)
{
        if(d == 2)
                aml_copy_t2d_helper(dst, cumul_dst_pitch, src, cumul_src_pitch, elem_number, elem_size);
        else {
                size_t * new_cumul_src_pitch = (size_t *)alloca((d-2)*sizeof(size_t));
                size_t * new_elem_number = (size_t *)alloca((d-1)*sizeof(size_t));
                // process dimension d-2
                for(int i = 0; i < d-3; i++)
                        new_cumul_src_pitch[i] = cumul_src_pitch[i];
                new_cumul_src_pitch[d-3] = cumul_src_pitch[d-2];
                for(int i = 0; i < d-2; i++)
                        new_elem_number[i] = elem_number[i];
                new_elem_number[d-2] = elem_number[d-1];
                for(int i = 0; i < elem_number[d-2]; i++) {
                        aml_copy_rtnd_helper(d-1, dst, cumul_dst_pitch, src, new_cumul_src_pitch, new_elem_number, elem_size);
                        dst += cumul_dst_pitch[d-2];
                        src += cumul_src_pitch[d-3];
                }
        }
}

int aml_copy_rtnd(size_t d, void *dst, const size_t *dst_pitch, const void *src, const size_t *src_pitch, const size_t *elem_number, const size_t elem_size)
{
        assert(d > 0);
        if(d == 1)
                memcpy(dst, src, elem_number[0]*elem_size);
        else {
                size_t * cumul_dst_pitch = (size_t *)alloca((d-1)*sizeof(size_t));
                size_t * cumul_src_pitch = (size_t *)alloca((d-1)*sizeof(size_t));
                assert(dst_pitch[0] >= elem_number[d-1]);
                assert(src_pitch[0] >= elem_number[0]);
                cumul_dst_pitch[0] = elem_size * dst_pitch[0];
                cumul_src_pitch[0] = elem_size * src_pitch[0];
                for(int i = 1; i < d - 1; i++) {
                        assert(dst_pitch[i] >= elem_number[i-1]);
                        assert(src_pitch[i] >= elem_number[i]);
                        cumul_dst_pitch[i] = dst_pitch[i] * cumul_dst_pitch[i-1];
                        cumul_src_pitch[i] = src_pitch[i] * cumul_src_pitch[i-1];
                }
                aml_copy_rtnd_helper(d, dst, cumul_dst_pitch, src, cumul_src_pitch, elem_number, elem_size);
        }
        return 0;
}

static void aml_copy_sh2d_helper(const size_t *target_dims, void *dst, const size_t *cumul_dst_pitch, const void *src, const size_t *cumul_src_pitch, const size_t *elem_number, const size_t elem_size)
{
        for(int j = 0; j < elem_number[1]; j++)
                for(int i = 0; i < elem_number[0]; i++)
                        memcpy(dst + i * cumul_dst_pitch[target_dims[0]] + j * cumul_dst_pitch[target_dims[1]],
                               src + i * cumul_src_pitch[0] + j * cumul_src_pitch[1],
                               elem_size);
}

static void aml_copy_shnd_helper(size_t d, const size_t *target_dims, void *dst, const size_t *cumul_dst_pitch, const void *src, const size_t *cumul_src_pitch, const size_t *elem_number, const size_t elem_size)
{
        if(d == 2)
                aml_copy_sh2d_helper(target_dims, dst, cumul_dst_pitch, src, cumul_src_pitch, elem_number, elem_size);
        else {
                // process dimension d-1
                for(int i = 0; i < elem_number[d-1]; i++) {
                        aml_copy_shnd_helper(d-1, target_dims, dst, cumul_dst_pitch, src, cumul_src_pitch, elem_number, elem_size);
                        dst += cumul_dst_pitch[target_dims[d-1]];
                        src += cumul_src_pitch[d-1];
                }
        }
}

int aml_copy_shnd(size_t d, const size_t *target_dims, void *dst, const size_t *dst_pitch, const void *src, const size_t *src_pitch, const size_t *elem_number, const size_t elem_size)
{
        assert(d > 0);
        if(d == 1)
                memcpy(dst, src, elem_number[0]*elem_size);
        else {
                size_t * cumul_dst_pitch = (size_t *)alloca(d*sizeof(size_t));
                size_t * cumul_src_pitch = (size_t *)alloca(d*sizeof(size_t));
                char * present_dims = (char *)alloca(d*sizeof(char));
                memset(present_dims, 0, d);
                cumul_dst_pitch[0] = elem_size;
                cumul_src_pitch[0] = elem_size;
                for(int i = 0; i < d; i++) {
                        assert(target_dims[i] < d);
                        if( target_dims[i] < d - 1 )
                                assert(dst_pitch[target_dims[i]] >= elem_number[i]);
                        present_dims[target_dims[i]] = 1;
                }
                for(int i = 0; i < d; i++)
                        assert(present_dims[i] == 1);
                for(int i = 0; i < d - 1; i++) {
                        assert(src_pitch[i] >= elem_number[i]);
                        cumul_dst_pitch[i+1] = dst_pitch[i] * cumul_dst_pitch[i];
                        cumul_src_pitch[i+1] = src_pitch[i] * cumul_src_pitch[i];
                }
                aml_copy_shnd_helper(d, target_dims, dst, cumul_dst_pitch, src, cumul_src_pitch, elem_number, elem_size);
        }
        return 0;
}
