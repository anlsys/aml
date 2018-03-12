#include <assert.h>
#include <aml.h>
#include <sys/mman.h>
#include <numaif.h>

/*******************************************************************************
 * mbind methods for Linux systems
 * Only handles the actual mbind/mempolicy calls
 ******************************************************************************/

/* common to both methods */
int aml_area_linux_mbind_generic_binding(struct aml_area_linux_mbind_data *data,
					 struct aml_binding **b)
{
	assert(data != NULL);
	/* not exactly proper, we should inspect the nodemask to find the real
	 * binding policy.
	 */
	if(data->policy == MPOL_BIND)
	{
		for(int i = 0; i < AML_MAX_NUMA_NODES; i++)
			if(AML_NODEMASK_ISSET(data->nodemask, i))
				return aml_binding_create(b, AML_BINDING_TYPE_SINGLE,i);
	}
	else if(data->policy == MPOL_INTERLEAVE)
	{
		return aml_binding_create(b, AML_BINDING_TYPE_INTERLEAVE,
					  data->nodemask);
	}
	return 0;
}


int aml_area_linux_mbind_regular_pre_bind(struct aml_area_linux_mbind_data *data)
{
	assert(data != NULL);
	return 0;
}

int aml_area_linux_mbind_regular_post_bind(struct aml_area_linux_mbind_data *data,
					   void *ptr, size_t sz)
{
	assert(data != NULL);
	return mbind(ptr, sz, data->policy, data->nodemask, AML_MAX_NUMA_NODES, 0);
}

struct aml_area_linux_mbind_ops aml_area_linux_mbind_regular_ops = {
	aml_area_linux_mbind_regular_pre_bind,
	aml_area_linux_mbind_regular_post_bind,
	aml_area_linux_mbind_generic_binding,
};

int aml_area_linux_mbind_setdata(struct aml_area_linux_mbind_data *data,
				 int policy, unsigned long *nodemask)
{
	assert(data != NULL);
	data->policy = policy;
	memcpy(data->nodemask, nodemask, AML_NODEMASK_BYTES);
	return 0;
}

int aml_area_linux_mbind_mempolicy_pre_bind(struct aml_area_linux_mbind_data *data)
{
	assert(data != NULL);
	/* function is called before mmap, we save the "generic" mempolicy into
	 * our data, and apply the one the user actually want
	 */
	int policy;
	unsigned long nodemask[AML_NODEMASK_SZ];
	int err;
	get_mempolicy(&policy, nodemask, AML_MAX_NUMA_NODES, NULL, 0);
	err = set_mempolicy(data->policy, data->nodemask, AML_MAX_NUMA_NODES);
	aml_area_linux_mbind_setdata(data, policy, nodemask);
	return err;
}

int aml_area_linux_mbind_mempolicy_post_bind(struct aml_area_linux_mbind_data *data,
					     void *ptr, size_t sz)
{
	assert(data != NULL);
	/* function is called after mmap, we retrieve the mempolicy we applied
	 * to it, and restore the generic mempolicy we saved earlier.
	 */
	int policy;
	unsigned long nodemask[AML_NODEMASK_SZ];
	int err;
	get_mempolicy(&policy, nodemask, AML_MAX_NUMA_NODES, NULL, 0);
	err = set_mempolicy(data->policy, data->nodemask, AML_MAX_NUMA_NODES);
	aml_area_linux_mbind_setdata(data, policy, nodemask);
	return err;
}

struct aml_area_linux_mbind_ops aml_area_linux_mbind_mempolicy_ops = {
	aml_area_linux_mbind_mempolicy_pre_bind,
	aml_area_linux_mbind_mempolicy_post_bind,
	aml_area_linux_mbind_generic_binding,
};

int aml_area_linux_mbind_init(struct aml_area_linux_mbind_data *data,
			      int policy, unsigned long *nodemask)
{
	assert(data != NULL);
	aml_area_linux_mbind_setdata(data, policy, nodemask);
	return 0;
}

int aml_area_linux_mbind_destroy(struct aml_area_linux_mbind_data *data)
{
	assert(data != NULL);
	return 0;
}
