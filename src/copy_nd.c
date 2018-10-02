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

static void aml_copy_nd_helper(size_t d, void *dst, const size_t *cumul_dst_pitch, const void *src, const size_t *cumul_src_pitch, const size_t *elem_number, const size_t elem_size) {
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

int aml_copy_tnd(size_t d, void *dst, const size_t *dst_pitch, const void *src, const size_t *src_pitch, const size_t *elem_number, const size_t elem_size)
{
        assert(d > 0);
        size_t *target_dims = (size_t *)alloca(d*sizeof(size_t));
        target_dims[0] = d - 1;
        for(int i = 1; i < d; i++)
                target_dims[i] = i-1;
        aml_copy_shnd(d, target_dims, dst, dst_pitch, src, src_pitch, elem_number, elem_size);
        return 0;
}

int aml_copy_rtnd(size_t d, void *dst, const size_t *dst_pitch, const void *src, const size_t *src_pitch, const size_t *elem_number, const size_t elem_size)
{
        assert(d > 0);
        size_t *target_dims = (size_t *)alloca(d*sizeof(size_t));
        target_dims[d-1] = 0;
        for(int i = 0; i < d-1; i++)
                target_dims[i] = i+1;
        aml_copy_shnd(d, target_dims, dst, dst_pitch, src, src_pitch, elem_number, elem_size);
        return 0;
}
