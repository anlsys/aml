#include <aml.h>
#include <assert.h>
#include <stdlib.h>

static inline int aml_copy_2d(void *dst, const size_t *cumul_dst_pitch, const void *src, const size_t *cumul_src_pitch, const size_t *elem_number, size_t elem_size)
{
        for(int i = 0; i < elem_number[1]; i++) {
                memcpy(dst, src, elem_number[0]*elem_size);
                dst += cumul_dst_pitch[0];
                src += cumul_src_pitch[0];
        }
        return 0;
}

static inline int aml_copy_nd_helper(size_t d, void *dst, const size_t *cumul_dst_pitch, const void *src, const size_t *cumul_src_pitch, const size_t *elem_number, const size_t elem_size) {
        if(d == 2)
                aml_copy_2d(dst, cumul_dst_pitch, src, cumul_src_pitch, elem_number, elem_size);
        else {
                for(int i = 0; i < elem_number[d-1]; i++) {
                        aml_copy_nd_helper(d-1, dst, cumul_dst_pitch, src, cumul_src_pitch, elem_number, elem_size);
                        dst += cumul_dst_pitch[d-2];
                        src += cumul_src_pitch[d-2];
                }
        }
        return 0;
}

int aml_copy_nd(size_t d, void *dst, const size_t *dst_pitch, const void *src, const size_t *src_pitch, const size_t *elem_number, const size_t elem_size)
{
        assert(d > 0);
        if(d == 1)
        {
                memcpy(dst, src, elem_number[0]*elem_size);
        } else {
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


static inline int aml_copy_t2d(void *dst, const size_t *dst_pitch, const void *src, const size_t *src_pitch, const size_t *elem_number, const size_t elem_size)
{
        assert( dst_pitch[0] >= elem_number[1] );
        assert( src_pitch[0] >= elem_number[0] );
        size_t dst_product_pitch = elem_size * dst_pitch[0];
        size_t src_product_pitch = elem_size * src_pitch[0];
        for(int j = 0; j < elem_number[1]; j++)
                for(int i = 0; i < elem_number[0]; i++)
                        memcpy(dst + j * elem_size + i * dst_product_pitch,
                               src + i * elem_size + j * src_product_pitch,
                               elem_size);
        return 0;
}

// d[j + i * dp[0]]
// d[j + k * dp[0] + i * dp[0] * dp[1]]
// d[j + k * dp[0] + l * dp[0] * dp[1] + i * dp[0] * dp[1] * dp[2]]
// s[i + j * sp[0]]
// s[i + j * sp[0] + k * sp[0] * sp[1]]
// s[i + j * sp[0] + k * sp[0] * sp[1] + l * sp[0] * sp[1] * sp[2]]
int aml_copy_tnd(size_t d, void *dst, const size_t *dst_pitch, const void *src, const size_t *src_pitch, const size_t *elem_number, const size_t elem_size)
{
        assert(d > 0);
        if(d == 1)
                memcpy(dst, src, elem_number[0]*elem_size);
        else if(d == 2)
                aml_copy_t2d(dst, dst_pitch, src, src_pitch, elem_number, elem_size);
        else {
                assert( dst_pitch[d-3] >= elem_number[d-2] );
                assert( src_pitch[d-2] >= elem_number[d-2] );

                size_t * new_dst_pitch = alloca((d-2)*sizeof(size_t));
                for(int i = 0; i < d-2; i++)
                        new_dst_pitch[i] = dst_pitch[i];
                new_dst_pitch[d-3] *= dst_pitch[d-2];
                size_t dst_product_pitch = elem_size;
                size_t src_product_pitch = elem_size;
                for(int i = 0; i < d-2; i++)
                        dst_product_pitch *= dst_pitch[i];
                for(int i = 0; i < d-1; i++)
                        src_product_pitch *= src_pitch[i];
                for(int i = 0; i < elem_number[d-1]; i++) {
                        aml_copy_tnd(d-1, dst, new_dst_pitch, src, src_pitch, elem_number, elem_size);
                        dst += dst_product_pitch;
                        src += src_product_pitch;
                }
        }
        return 0;
}

// d[l + i * dp[0]]
// d[l + i * dp[0] + k * dp[0] * dp[1]]
// d[l + i * dp[0] + j * dp[0] * dp[1] + k * dp[0] * dp[1] * dp[2]]
// s[i + l * sp[0]]
// s[i + k * sp[0] + l * sp[0] * sp[1]]
// s[i + j * sp[0] + k * sp[0] * sp[1] + l * sp[0] * sp[1] * sp[2]]
int aml_copy_rtnd(size_t d, void *dst, const size_t *dst_pitch, const void *src, const size_t *src_pitch, const size_t *elem_number, const size_t elem_size)
{
        assert(d > 0);
        if(d == 1)
                memcpy(dst, src, elem_number[0]*elem_size);
        else if(d == 2)
                aml_copy_t2d(dst, dst_pitch, src, src_pitch, elem_number, elem_size);
        else {
                assert( src_pitch[1] >= elem_number[1] );

                size_t * new_dst_pitch = alloca((d-2)*sizeof(size_t));
                size_t * new_src_pitch = alloca((d-2)*sizeof(size_t));
                size_t * new_elem_number = alloca((d-1)*sizeof(size_t));
                new_dst_pitch[0] = dst_pitch[0];
                for(int i = 1; i < d-2; i++)
                        new_dst_pitch[i] = dst_pitch[i+1];
                new_dst_pitch[1] *= dst_pitch[1];
                for(int i = 0; i < d-2; i++)
                        new_src_pitch[i] = src_pitch[i+1];
                new_src_pitch[0] *= src_pitch[0];
                new_elem_number[0] = elem_number[0];
                for(int i = 1; i < d-1; i++)
                        new_elem_number[i] = elem_number[i+1];
                size_t dst_product_pitch = elem_size*dst_pitch[0]*dst_pitch[1];
                size_t src_product_pitch = elem_size*src_pitch[0];
                for(int i = 0; i < elem_number[1]; i++) {
                        aml_copy_rtnd(d-1, dst, new_dst_pitch, src, new_src_pitch, new_elem_number, elem_size);
                        dst += dst_product_pitch;
                        src += src_product_pitch;
                }
        }
        return 0;
}
