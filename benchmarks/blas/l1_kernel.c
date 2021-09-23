/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/
/*
 * This is a benchmark for the BLAS Level 1 operations for AML.
 */

#include "blas/l1_kernel.h"

/* Look into another way to define these */
#define sign(a) ((a > 0) ? 1 : ((a < 0) ? -1 : 0))

double dasum(size_t n, double *a, double *b, double *c, double scalar)
{
	(void)*b;
	(void)*c;
	(void)scalar;
	size_t i;
	double dasum = 0;

	for (i = 0; i < n; i++) {
		dasum = dasum + fabs(a[i]);
	}
	return dasum;
}

double daxpy(size_t n, double *a, double *b, double *c, double scalar)
{
	size_t i;
#pragma omp parallel for
	for (i = 0; i < n; i++)
		c[i] = b[i] + scalar * a[i];
	return 1;
}

double dcopy(size_t n, double *a, double *b, double *c, double scalar)
{
	(void)*c;
	(void)scalar;
	size_t i;

#pragma omp parallel for
	for (i = 0; i < n; i++)
		b[i] = a[i];
	return 1;
}

double ddot(size_t n, double *a, double *b, double *c, double scalar)
{
	(void)*c;
	(void)scalar;
	size_t i;
	long double dot = 0.0;

#pragma omp parallel for reduction(+ : dot)
	for (i = 0; i < n; i++) {
		long double temp;
		temp = a[i] * b[i];
		dot += temp;
	}
	return (double)dot;
}

double dnrm2(size_t n, double *a, double *b, double *c, double scalar)
{
	(void)*b;
	(void)*c;
	(void)scalar;
	size_t i;
	double scale, ssq, temp;

	scale = 0.0;
	ssq = 1.0;
	for (i = 0; i < n; i++) {
		if (a[i] != 0.0) {
			temp = fabs(a[i]);
			if (scale < temp) {
				ssq = 1.0 + ssq * pow(scale / temp, 2);
				scale = temp;
			} else
				ssq = ssq + pow(temp / scale, 2);
		}
	}
	return scale * sqrt(ssq);
}

double dscal(size_t n, double *a, double *b, double *c, double scalar)
{
	(void)*c;
	size_t i;

#pragma omp parallel for
	for (i = 0; i < n; i++)
		b[i] = scalar * a[i];
	return 1;
}

double dswap(size_t n, double *a, double *b, double *c, double scalar)
{
	(void)*c;
	(void)scalar;
	size_t i;

#pragma omp parallel for
	for (i = 0; i < n; i++) {
		double temp = a[i];
		a[i] = b[i];
		b[i] = temp;
	}
	return 1;
}

double idmax(size_t n, double *a, double *b, double *c, double scalar)
{
	(void)*b;
	(void)*c;
	(void)scalar;

	if (n == 1)
		return 0;

	size_t i;
	double max;
	size_t id_max = 0;

	max = a[0];
	for (i = 1; i < n; i++) {
		if (fabs(a[i]) > max) {
			id_max = i;
			max = fabs(a[i]);
		}
	}
	return id_max;
}

/* The rotations. Not included in the array of functions because of their
   parameters */
/* Plane rotation */
void drot(size_t n, double *a, double *b, double x, double y)
{
	size_t i;

#pragma omp parallel for
	for (i = 0; i < n; i++) {
		double temp = x * a[i] + y * b[i];
		b[i] = x * b[i] - y * a[i];
		a[i] = temp;
	}
}

/* Create a plane rotation. TODO: Verify */
void drotg(double x, double y, double c, double s)
{
	double r, roe, scale, z;

	roe = y;
	if (fabs(x) > fabs(y))
		roe = x;
	scale = fabs(x) + fabs(y);
	if (scale == 0.0) {
		c = 1.0;
		s = 0.0;
		r = 0.0;
		z = 0.0;
	} else {
		r = scale * sqrt(pow(x / scale, 2) + pow(y / scale, 2));
		r = sign(roe) * r;
		c = x / r;
		s = y / r;
		z = 1.0;
		if (fabs(x) > fabs(y))
			z = s;
		if (fabs(y) >= fabs(x) && c != 0.0)
			z = 1.0 / c;
	}
	x = r;
	y = z;
}

void drotm(size_t n, double *a, double *b, double *param)
{
	double flag, h11, h12, h21, h22;
	size_t i;

	flag = param[0];
	if (flag < 0.0) {
		h11 = param[1];
		h12 = param[3];
		h21 = param[2];
		h22 = param[4];
	} else {
		if (flag == 0) {
			h11 = 1.0;
			h12 = param[3];
			h21 = param[2];
			h22 = 1.0;
		} else {
			h11 = param[1];
			h12 = 1.0;
			h21 = -1.0;
			h22 = param[4];
		}
	}

#pragma omp parallel for
	for (i = 0; i < n; i++) {
		double w = a[i];
		double z = b[i];
		a[i] = w * h11 + z * h12;
		b[i] = w * h21 + z * h22;
	}
}

/* TODO: Verify */
void drotmg(double d1, double d2, double x, double y, double *param)
{
	double flag, h11, h12, h21, h22, p1, p2, q1, q2, temp, u, gam, gamsq,
	        rgamsq;

	gam = 4096.0;
	gamsq = 16777216.0;
	rgamsq = 5.9604645e-8;

	/* default initialization */
	h11 = 0.0;
	h12 = 0.0;
	h21 = 0.0;
	h22 = 0.0;

	if (d1 < 0) {
		flag = -1.0;
		d1 = 0.0;
		d2 = 0.0;
		x = 0.0;
	} else {
		p2 = d2 * y;
		if (p2 == 0) {
			flag = -2.0;
			param[0] = flag;
		}
		p1 = d1 * x;
		q2 = p2 * y;
		q1 = p1 * x;

		if (fabs(q1) > fabs(q2)) {
			h21 = -y / x;
			h12 = p2 / p1;
			u = 1.0 - h12 * h21;

			if (u > 0) {
				flag = 0.0;
				d1 = d1 / u;
				d2 = d2 / u;
				x = x * u;
			}
		} else {
			if (q2 < 0.0) {
				flag = -1.0;
				d1 = 0.0;
				d2 = 0.0;
				x = 0.0;
			} else {
				flag = 1.0;
				h11 = p1 / p2;
				h22 = x / y;
				u = 1.0 + h11 * h22;
				temp = d2 / u;
				d2 = d1 / u;
				d1 = temp;
				x = y * u;
			}
		}
		if (d1 != 0.0) {
			while (fabs(d1) <= rgamsq || d1 >= gamsq) {
				if (flag == 0.0) {
					h11 = 1.0;
					h22 = 1.0;
				} else {
					h21 = -1.0;
					h12 = 1.0;
				}
				flag = -1.0;
				if (d1 <= rgamsq) {
					d1 = d1 * pow(gam, 2);
					x = x / gam;
					h11 = h11 / gam;
					h12 = h12 / gam;
				} else {
					d1 = d1 / pow(gam, 2);
					x = x * gam;
					h11 = h11 * gam;
					h12 = h12 * gam;
				}
			}
		}
		if (d2 != 0) {
			while (fabs(d2) <= rgamsq || fabs(d2) >= gamsq) {
				if (flag == 0.0) {
					h11 = 1.0;
					h22 = 1.0;
				} else {
					h21 = -1.0;
					h12 = 1.0;
				}
				flag = -1.0;
				if (fabs(d2) <= rgamsq) {
					d2 = d2 * pow(gam, 2);
					h21 = h21 / gam;
					h22 = h22 / gam;
				} else {
					d2 = d2 / pow(gam, 2);
					h21 = h21 * gam;
					h22 = h22 * gam;
				}
			}
		}
	}
	param[1] = h11;
	param[2] = h21;
	param[3] = h12;
	param[4] = h22;
	param[0] = flag;
}
