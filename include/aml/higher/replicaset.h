/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef __AML_HIGHER_REPLICASET_H_
#define __AML_HIGHER_REPLICASET_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @}
 * @defgroup aml_replicaset "AML Replicaset"
 * @brief Maintain a copy of a data in several areas.
 *
 * Replicaset is a building block on top of areas to
 * maintain a manage coherency of several copies of a data in several areas.
 * @{
 **/

// Implementation specific data held in abstract replicaset.
struct aml_replicaset_data;

// Replicaset required methods. See below
struct aml_replicaset_ops;

/**
 * High level replicaset structure.
 * See specific implementations for instanciation.
 */
struct aml_replicaset {
	/** Number of replicas **/
	unsigned int n;
	/** Size of a single replica **/
	size_t size;
	/** Pointers to replicated data **/
	void **replica;
	/** Replicaset methods **/
	struct aml_replicaset_ops *ops;
	/** Replicaset implementation spepcific data **/
	struct aml_replicaset_data *data;
};

/**
 * aml_replicaset_ops is a structure containing implementations
 * of replicaset operations.
 * Users may create or modify implementations by assembling
 * appropriate operations in such a structure.
 **/
struct aml_replicaset_ops {
	/**
	 * Initialize replicas of a replicaset with some data.
	 * @param replicaset[in]: The replicaset holding copies.
	 * @param data[in]: The data to copy.
	 * @return An AML error code.
	 **/
	int (*init)(struct aml_replicaset *replicaset, const void *data);

	/**
	 * Copy the content of a specific replica into other replicas.
	 * @param replicaset[in]: The replicaset holding copies.
	 * @param id[in]: The index of the replica holding data to copy.
	 * @return An AML error code.
	 **/
	int (*sync)(struct aml_replicaset *replicaset, const unsigned int id);
};

/**
 * @see struct aml_replicaset_ops->init()
 */
int aml_replicaset_init(struct aml_replicaset *replicaset, const void *data);

/**
 * @see struct aml_replicaset_ops->sync()
 */
int aml_replicaset_sync(struct aml_replicaset *replicaset,
                        const unsigned int id);

#ifdef __cplusplus
}
#endif

#endif // __AML_HIGHER_REPLICASET_H_
