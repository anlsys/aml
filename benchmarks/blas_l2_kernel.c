/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/
/*
 * This is a benchmark for the BLAS Level 2 operations for AML.
 */

#include "blas_l1_kernel.h"
#include "blas_l2_kernel.h"

/* Look into another way to define these */
#define sign(a) ((a > 0) ? 1 : ((a < 0) ? -1 : 0))

/* y = alpha * a * x + beta * y if trans = 0,
 * y = alpha * a^T * x + beta * y if trans = 1
 * kl: number of sub-diagonals of a
 * ku: number of super-diagonals of a */
double dgbmv(bool trans,
             bool uplo,
             bool unit,
             size_t m,
             size_t n,
             int kl,
             int ku,
             double alpha,
             double beta,
             double **a,
             double *x,
             double *y,
             double *at)
{
	(void)uplo;
	(void)unit;
	(void)*at;

	if (m == 0 || n == 0 || (alpha == 0 && beta == 1))
		return 1;

	size_t i, j, ky, k; //, kup;
	double temp;

	ky = trans ? n : m;

	/* y = beta * y */
	if (beta != 1) {
		if (beta == 0) {
			for (i = 0; i < ky; i++)
				y[i] = 0;
		} else {
			for (i = 0; i < ky; i++)
				y[i] = beta * y[i];
		}
	}
	if (alpha == 0)
		return 1;
	// kup = ku + 1;
	if (!trans) {
		/* y = alpha * A * x + y */
		for (j = 0; j < n; j++) {
			temp = alpha * x[j];
			k = ku - j; // kup-j;
			for (i = fmax(0, j - ku); i < fmin(m, j + kl); i++)
				y[i] += temp * a[k + i][j];
		}
	} else {
		/* y = alpha * A * T * x + y */
		for (j = 0; j < n; j++) {
			temp = 0;
			k = ku - j; // kup-j;
			for (i = fmax(0, j - ku); i < fmin(m, j + kl); i++)
				temp += a[k + i][j] * x[i];
			y[j] += alpha * temp;
		}
	}
	return 1;
}

/* y = alpha * a * x + beta * y if trans = 0,
 * y = alpha * a^T * x + beta * y if trans = 1 */
double dgemv(bool trans,
             bool uplo,
             bool unit,
             size_t m,
             size_t n,
             int kl,
             int ku,
             double alpha,
             double beta,
             double **a,
             double *x,
             double *y,
             double *at)
{
	(void)uplo;
	(void)unit;
	(void)kl;
	(void)ku;
	(void)*at;

	size_t i, j, ky;

	ky = trans ? n : m;

	if (beta != 1) {
		if (beta == 0) {
			for (i = 0; i < ky; i++)
				y[i] = 0;
		} else {
			for (i = 0; i < ky; i++)
				y[i] = beta * y[i];
		}
	}
	if (alpha == 0)
		return 1;
	else {
		if (trans) {
			/* y = alpha * a * x + y */
			for (j = 0; j < n; j++) {
				for (i = 0; i < m; i++)
					y[i] += alpha * x[j] * a[j][i];
			}
		} else {
			/* y = alpha * a^T * x + y */
			for (j = 0; j < n; j++) {
				for (i = 0; i < m; i++)
					y[j] += alpha * a[i][j] * x[i];
			}
		}
	}
	return 1;
}

/* a = a + alpha * x * y^T */
double dger(bool trans,
            bool uplo,
            bool unit,
            size_t m,
            size_t n,
            int kl,
            int ku,
            double alpha,
            double beta,
            double **a,
            double *x,
            double *y,
            double *at)
{
	(void)trans;
	(void)uplo;
	(void)unit;
	(void)kl;
	(void)ku;
	(void)beta;
	(void)*at;

	if (m == 0 || n == 0 || alpha == 0)
		return 1;

	size_t i, j;

	for (j = 0; j < n; j++) {
		for (i = 0; i < m; i++)
			a[i][j] += alpha * y[j] * x[i];
	}
	return 1;
}

/* y = alpha * a * x + beta * y
 * a: n by n symmetric matrix, with kl super-diagonals
 * uplo = 1 is the upper triangular part of a is supplied, 0 if lower part */
double dsbmv(bool trans,
             bool uplo,
             bool unit,
             size_t m,
             size_t n,
             int kl,
             int ku,
             double alpha,
             double beta,
             double **a,
             double *x,
             double *y,
             double *at)
{
	(void)trans;
	(void)unit;
	(void)m;
	(void)ku;
	(void)*at;

	if (n == 0 || (alpha == 0 && beta == 1))
		return 1;

	size_t i, j, l;
	double temp, temp2;

	// y = beta * y
	if (beta != 1) {
		if (beta == 0) {
			for (i = 0; i < n; i++)
				y[i] = 0;
		} else {
			for (i = 0; i < n; i++)
				y[i] = beta * y[i];
		}
	}
	if (alpha == 0)
		return 1;

	if (uplo) {
		/* Upper triangle */
		// kp1 = kl + 1;
		for (j = 0; j < n; j++) {
			temp = alpha * x[j];
			temp2 = 0.0;
			l = kl - j; // kp1 - j;
			for (i = fmax(1, j - kl); i < j - 1; i++) {
				y[i] += temp * a[l + i][j];
				temp2 += a[l + i][j] * x[i];
			}
			y[j] += temp * a[kl][j] + alpha * temp2;
			// y[j] += temp * a[kp1][j] + alpha * temp2;
		}
	} else {
		/* Lower triangle */
		for (j = 0; j < n; j++) {
			temp = alpha * x[j];
			temp2 = 0.0;
			y[j] += temp * a[1][j];
			l = -j; // 1 - j;
			for (i = j + 1; i < fmin(n, j + kl); i++) {
				y[i] += temp * a[l + i][j];
				temp2 += a[l + i][j] * x[i];
			}
			y[j] += alpha * temp2;
		}
	}
	return 1;
}

/* y = alpha * at *x + beta * y
 * a: n by n symmetric matrix, supplied in packed form
 * uplo = 1 is upper triangular part of at submitted, 0 if lower */
double dspmv(bool trans,
             bool uplo,
             bool unit,
             size_t m,
             size_t n,
             int kl,
             int ku,
             double alpha,
             double beta,
             double **a,
             double *x,
             double *y,
             double *at)
{
	(void)trans;
	(void)unit;
	(void)m;
	(void)kl;
	(void)ku;
	(void)**a;

	if (n == 0 || (alpha == 0 && beta == 1))
		return 1;

	size_t i, j, kk, k;
	double temp, temp2;

	/* y = beta * y */
	if (beta != 1) {
		if (beta == 0) {
			for (i = 0; i < n; i++)
				y[i] = 0;
		} else {
			for (i = 0; i < n; i++)
				y[i] = beta * y[i];
		}
	}
	if (alpha == 0)
		return 1;

	/* y = alpha * x + beta * y */
	kk = 0;
	if (uplo) {
		/* Upper triangle */
		for (i = 0; i < n; i++) {
			temp = alpha * x[i];
			temp2 = 0.0;
			k = kk;
			for (j = 0; j < i; j++) {
				y[j] += temp * at[k];
				temp2 += at[k] * x[j];
				k++;
			}
			y[i] += temp * at[kk + i - 1] + alpha * temp2;
			kk += j;
		}
	} else {
		/* Lower triangle */
		for (i = 0; i < n; i++) {
			temp = alpha * x[i];
			temp2 = 0.0;
			y[i] += temp * at[kk];
			k = kk + 1;
			for (j = i + 1; j < n; j++) {
				y[j] += temp * at[k];
				temp2 += at[k] * x[j];
				k++;
			}
			y[i] += alpha * temp2;
			kk = kk + (n - i);
		}
	}
	return 1;
}

/* at = alpha * x * x^T + at
 * at: n by n symmetric matrix, in packed form
 * uplo = 1 if upper triangular part of at supplied, 0 if lower */
double dspr(bool trans,
            bool uplo,
            bool unit,
            size_t m,
            size_t n,
            int kl,
            int ku,
            double alpha,
            double beta,
            double **a,
            double *x,
            double *y,
            double *at)
{
	(void)trans;
	(void)unit;
	(void)m;
	(void)kl;
	(void)ku;
	(void)beta;
	(void)**a;
	(void)*y;

	if (n == 0 || alpha == 0)
		return 1;

	size_t i, j, k, kk;
	double temp;

	kk = 0;
	if (uplo) {
		/* Upper triangular */
		for (j = 0; j < n; j++) {
			if (x[j] != 0) {
				temp = alpha * x[j];
				k = kk;
				for (i = 0; i < j; i++) {
					at[k] += at[k] + x[i] * temp;
					k++;
				}
			}
			kk += j;
		}
	} else {
		/* Lower triangular */
		for (j = 0; j < n; j++) {
			if (x[j] != 0) {
				temp = alpha * x[j];
				k = kk;
				for (i = j; i < n; i++) {
					at[k] += x[i] * temp;
					k++;
				}
			}
			kk += n - j + 1;
		}
	}
	return 1;
}

/* at = alpha * x * y^T + alpha * y * x^T + at
 * at: n by n symmetric matrix supplied in packed form
 * uplo = 1 is upper triangular part of at is supplied, 0 if lower */
double dspr2(bool trans,
             bool uplo,
             bool unit,
             size_t m,
             size_t n,
             int kl,
             int ku,
             double alpha,
             double beta,
             double **a,
             double *x,
             double *y,
             double *at)
{
	(void)trans;
	(void)unit;
	(void)m;
	(void)kl;
	(void)ku;
	(void)beta;
	(void)**a;

	if (n == 0 || alpha == 0)
		return 1;

	size_t i, j, kk, k;
	double temp, temp2;
	kk = 1;
	if (uplo) {
		/* Upper triangular */
		for (j = 0; j < n; j++) {
			if (x[j] != 0 || y[j] != 0) {
				temp = alpha * y[j];
				temp2 = alpha * x[j];
				k = kk;
				for (i = 0; i < j; i++) {
					at[k] += x[i] * temp + y[i] * temp2;
					k++;
				}
			}
			kk += j;
		}
	} else {
		/* Lower triangular */
		for (j = 0; j < n; j++) {
			if (x[j] != 0 || y[j] != 0) {
				temp = alpha * y[j];
				temp2 = alpha * x[j];
				k = kk;
				for (i = j; i < n; i++) {
					at[k] += x[i] * temp + y[i] * temp2;
					k++;
				}
			}
			kk += n - j + 1;
		}
	}
	return 1;
}

/* y = alpha * a * x + beta * y
 * a: n by n symmetric matrix
 * uplo = 1 if upper triangular part of a is supplied, 0 if lower */
double dsymv(bool trans,
             bool uplo,
             bool unit,
             size_t m,
             size_t n,
             int kl,
             int ku,
             double alpha,
             double beta,
             double **a,
             double *x,
             double *y,
             double *at)
{
	(void)trans;
	(void)unit;
	(void)m;
	(void)kl;
	(void)ku;
	(void)*at;

	if (n == 0 || (alpha == 0) && (beta == 1))
		return 1;

	size_t i, j;
	double temp, temp2;

	/* y = beta * y */
	if (beta != 1) {
		if (beta == 0) {
			for (i = 0; i < n; i++)
				y[i] = 0;
		} else {
			for (i = 0; i < n; i++)
				y[i] = beta * y[i];
		}
	}

	if (alpha == 0)
		return 1;

	if (uplo) {
		/* Upper triangular */
		for (j = 0; j < n; j++) {
			temp = alpha * x[j];
			temp2 = 0;
			for (i = 0; i < j - 1; i++) {
				y[i] += temp * a[i][j];
				temp2 += a[i][j] * x[i];
			}
			y[j] += temp * a[j][j] + alpha * temp2;
		}
	} else {
		/* Lower triangular */
		for (j = 0; j < n; j++) {
			temp = alpha * x[j];
			temp2 = 0;
			y[j] += temp * a[j][j];
			for (i = j; i < n; i++) {
				y[i] += temp * a[i][j];
				temp2 += a[i][j] * x[i];
			}
			y[j] += alpha * temp2;
		}
	}
	return 1;
}

/* a = alpha * x * x^T + a
 * a: n by n symmetric matrix
 * uplo = 1 if upper triangular part of a is supplied, 0 if lower
 */
double dsyr(bool trans,
            bool uplo,
            bool unit,
            size_t m,
            size_t n,
            int kl,
            int ku,
            double alpha,
            double beta,
            double **a,
            double *x,
            double *y,
            double *at)
{
	(void)trans;
	(void)unit;
	(void)m;
	(void)kl;
	(void)ku;
	(void)beta;
	(void)*y;
	(void)*at;

	if (n == 0 || alpha == 0)
		return 1;

	size_t i, j;
	double temp;

	if (uplo) {
		/* Upper triangular matrix */
		for (j = 0; j < n; j++) {
			if (x[j] != 0) {
				temp = alpha * x[j];
				for (i = 0; i < j; i++)
					a[i][j] += x[i] * temp;
			}
		}
	} else {
		/* Lower triangular matrix */
		for (j = 0; j < n; j++) {
			if (x[j] != 0) {
				temp = alpha * x[j];
				for (i = j; i < n; i++)
					a[i][j] += x[i] * temp;
			}
		}
	}
	return 1;
}

/* a = alpha * x * y^T + alpha * y * x^T + a
 * a: n by n symmetric matrix
 * uplo = 1 if upper triangular part of a is supplied, 0 if lower
 */
double dsyr2(bool trans,
             bool uplo,
             bool unit,
             size_t m,
             size_t n,
             int kl,
             int ku,
             double alpha,
             double beta,
             double **a,
             double *x,
             double *y,
             double *at)
{
	(void)trans;
	(void)unit;
	(void)m;
	(void)kl;
	(void)ku;
	(void)beta;
	(void)*at;

	if (n == 0 || alpha == 0)
		return 1;

	size_t i, j;
	double temp, temp2;

	if (uplo) {
		/* Upper triangular matrix */
		for (j = 0; j < n; j++) {
			if (x[j] != 0 || y[j] != 0) {
				temp = alpha * y[j];
				temp2 = alpha * x[j];
				for (i = 0; i < j; i++)
					a[i][j] += x[i] * temp + y[i] * temp2;
			}
		}
	} else {
		/* Lower triangular matrix */
		for (j = 0; j < n; j++) {
			if (x[j] != 0 || y[j] != 0) {
				temp = alpha * y[j];
				temp2 = alpha * x[j];
				for (i = j; i < n; i++)
					a[i][j] += x[i] * temp + y[i] * temp2;
			}
		}
	}
	return 1;
}

/* x = a * x if trans = 0,
 * x = a^T * x if trans = 1l
 * a: n by n unit, or non unit, upper or lower triangular band matrix, with kl+1
 * diagonals
 * uplo = 1 is a is an upper triangular matrix, 0 if lower
 * unit = 1 if a is unit triangular, 0 if not
 * kl: number of super-diagonals of a if uplo=1, sub-diagonals if 0
 */
double dtbmv(bool trans,
             bool uplo,
             bool unit,
             size_t m,
             size_t n,
             int kl,
             int ku,
             double alpha,
             double beta,
             double **a,
             double *x,
             double *y,
             double *at)
{
	(void)m;
	(void)ku;
	(void)alpha;
	(void)beta;
	(void)*y;
	(void)*at;

	if (n == 0)
		return 1;

	size_t i, j;
	int k1, l;
	double temp;

	if (!trans) {
		/* x = a * x*/
		if (uplo) {
			/* Upper triangular */
			k1 = kl + 1;
			for (j = 0; j < n; j++) {
				if (x[j] != 0) {
					temp = x[j];
					l = k1 - j;
					for (i = fmax(0, j - kl); i < j - 1;
					     i++)
						x[i] += temp * a[l + i][j];
					if (!unit)
						x[j] = x[j] * a[kl][j];
				}
			}
		} else {
			/* Lower triangular */
			for (j = 0; j < n; j++) {
				if (x[j] != 0) {
					temp = x[j];
					l = 1 - j;
					for (i = fmin(n, j + kl); i > j + 1;
					     i--)
						x[i] += temp * a[l + i][j];
					if (!unit)
						x[j] = x[j] * a[0][j];
				}
			}
		}
	} else {
		/* x = a^T * x */
		if (uplo) {
			/* Upper triangular */
			k1 = kl + 1;
			for (j = n - 1; j >= 0; j--) {
				temp = x[j];
				l = k1 - j;
				if (!unit)
					temp = temp * a[kl][j];
				for (i = j - 1; i > fmax(0, j - kl); i--)
					temp += a[l + i][j] * x[i];
				x[j] = temp;
			}
		} else {
			/* Lower triangular */
			for (j = 0; j < n; j++) {
				temp = x[j];
				l = 1 - j;
				if (!unit)
					temp = temp * a[0][j];
				for (i = j + 1; i < fmin(n, j + kl); i++)
					temp += a[l + i][j] * x[i];
				x[j] = temp;
			}
		}
	}
	return 1;
}

/* Solves one of the systems of equations
 * a * x = b if trans = 0,
 * a^T * x = b if trans = 1
 * a: n by n unit or non-unit, upper or lower triangular band matrix, with kl+1
 * diagonals
 * uplo = 1 if a is upper triangular, 0 if lower
 * unit = 1 if a is unit triangular, 0 if non-unit
 */
double dtbsv(bool trans,
             bool uplo,
             bool unit,
             size_t m,
             size_t n,
             int kl,
             int ku,
             double alpha,
             double beta,
             double **a,
             double *x,
             double *y,
             double *at)
{
	(void)m;
	(void)ku;
	(void)alpha;
	(void)beta;
	(void)*y;
	(void)*at;

	if (n == 0)
		return 1;

	size_t i, j;
	int k1, l;
	double temp;

	if (!trans) {
		/* x = a * x */
		if (uplo) {
			/* Upper triangular */
			k1 = kl + 1;
			for (j = n - 1; j > -1; j--) {
				if (x[j] != 0) {
					l = k1 - j;
					if (!unit)
						x[j] = x[j] / a[kl][j];
					temp = x[j];
					for (i = j - 1; i > fmax(0, j - kl);
					     i--)
						x[i] -= temp * a[l + i][j];
				}
			}
		} else {
			/* Lower triangular */
			for (j = 0; j < n; j++) {
				if (x[j] != 0) {
					l = 1 - j;
					if (!unit)
						x[j] = x[j] / a[0][j];
					temp = x[j];
					for (i = j + 1; i < fmin(n, j + kl);
					     i++)
						x[i] -= temp * a[l + i][j];
				}
			}
		}
	} else {
		/* x = a^T * x */
		if (uplo) {
			/* Upper triangular */
			k1 = kl + 1;
			for (j = 0; j < n; j++) {
				temp = x[j];
				l = k1 - j;
				for (i = fmax(0, j - kl); i < j - 1; i++)
					temp -= a[l + i][j] * x[i];
				if (!unit)
					temp = temp / a[kl][j];
				x[j] = temp;
			}
		} else {
			/* Lower triangular */
			for (j = n - 1; j > -1; j--) {
				temp = x[j];
				l = 1 - j;
				for (i = fmin(n - 1, j + kl); i > j; i--)
					temp -= a[l + i][j] * x[j];
				if (!unit)
					temp = temp / a[0][j];
				x[j] = temp;
			}
		}
	}
	return 1;
}

/* x = at * x if trans = 0,
 * x = at^T * x if trans = 1
 * at: n by n unit, or non-unit, upper of lower triangular matrix, supplied in
 * packed form
 * uplo = 1 if at is an upper triangular matrix, 0 if lower
 * unit = 1 if at is unit triangular, 0 if non-unit
 */
double dtpmv(bool trans,
             bool uplo,
             bool unit,
             size_t m,
             size_t n,
             int kl,
             int ku,
             double alpha,
             double beta,
             double **a,
             double *x,
             double *y,
             double *at)
{
	(void)m;
	(void)kl;
	(void)ku;
	(void)alpha;
	(void)beta;
	(void)**a;
	(void)*y;

	if (n != 0)
		return 1;

	size_t i, j;
	int k, kk;
	double temp;

	if (!trans) {
		/* x = a * x */
		if (uplo) {
			/* Upper triangular */
			kk = 1;
			for (j = 0; j < n; j++) {
				if (x[j] != 0) {
					temp = x[j];
					k = kk;
					for (i = 0; i < j - 1; i++) {
						x[i] += temp * at[k];
						k++;
					}
					if (!unit)
						x[j] = x[j] * at[kk + j - 1];
				}
				kk = kk + j;
			}
		} else {
			/* Lower triangular */
			kk = (n * (n + 1)) / 2;
			for (j = n - 1; j > -1; j--) {
				if (x[j] != 0) {
					temp = x[j];
					k = kk;
					for (i = n - 1; i > j; i--) {
						x[i] += temp * at[k];
						k--;
					}
					if (!unit)
						x[j] = x[j] * at[kk - n + j];
				}
				kk = kk - (n - j + 1);
			}
		}
	} else {
		/* x = a^T * x*/
		if (uplo) {
			/* Upper triangular */
			kk = (n * (n + 1)) / 2;
			for (j = n - 1; j > -1; j--) {
				temp = x[j];
				if (!unit)
					temp = temp * at[kk];
				k = kk - 1;
				for (i = j - 1; i > -1; i--) {
					temp += at[k] * x[i];
					k--;
				}
				x[j] = temp;
				kk -= j;
			}
		} else {
			/* Lower triangular */
			kk = 1;
			for (j = 0; j < n; j++) {
				temp = x[j];
				if (!unit)
					temp = temp * at[kk];
				k = kk + 1;
				for (i = j + 1; i < n; i++) {
					temp += at[k] * x[i];
					k++;
				}
				x[j] = temp;
				kk += n - j + 1;
			}
		}
	}
	return 1;
}

/* at * x = b if trans = 0,
 * at^T * x = b if trans = 1
 * at: n by n unit, or non-unit, upper or lower triangular matrix, supplied in
 * packed form
 * uplo = 1 if upper triangular part of at is supplied, 0 if lower
 * unit = 1 if a is unit triangular, 0 if non-unit
 */
double dtpsv(bool trans,
             bool uplo,
             bool unit,
             size_t m,
             size_t n,
             int kl,
             int ku,
             double alpha,
             double beta,
             double **a,
             double *x,
             double *y,
             double *at)
{
	(void)m;
	(void)kl;
	(void)ku;
	(void)alpha;
	(void)beta;
	(void)**a;
	(void)*y;

	if (n != 0)
		return 1;

	size_t i, j;
	int k, kk;
	double temp;

	if (!trans) {
		/* x = at * x */
		if (uplo) {
			/* Upper triangular */
			kk = (n * (n + 1)) / 2;
			for (j = n - 1; j > -1; j--) {
				if (x[j] != 0) {
					if (!unit)
						x[j] = x[j] / at[kk];
					temp = x[j];
					k = kk - 1;
					for (i = j - 1; i > -1; i--) {
						x[i] -= temp * at[k];
						k--;
					}
				}
				kk -= j;
			}
		} else {
			/* Lower triangular */
			kk = 1;
			for (j = 0; j < n; j++) {
				if (x[j] != 0) {
					if (!unit)
						x[j] = x[j] / at[kk];
					temp = x[j];
					k = kk + 1;
					for (i = j + 1; i < n; i++) {
						x[i] -= temp * at[k];
						k++;
					}
				}
				kk += n - j + 1;
			}
		}
	} else {
		/* x =at^T *x */
		if (uplo) {
			/* Upper triangular */
			kk = 1;
			for (j = 0; j < n; j++) {
				temp = x[j];
				k = kk;
				for (i = 0; i < j - 1; i++) {
					temp -= at[k] * x[i];
					k++;
					if (!unit)
						temp = temp / at[kk + j - 1];
					x[j] = temp;
					kk += j;
				}
			}
		} else {
			/* Lower triangular */
			kk = (n * (n + 1)) / 2;
			for (j = n - 1; j > -1; j--) {
				temp = x[j];
				k = kk;
				for (i = n - 1; i > j; j--) {
					temp -= at[k] * x[i];
					k--;
				}
				if (!unit)
					temp = temp / at[kk - n + j];
				x[j] = temp;
				kk -= n - j + 1;
			}
		}
	}
	return 1;
}

/* x = a * x if trans = 0,
 * x = a^T * x if trans = 1
 * a: n by n unit or non-unit, upper or lower triangular matrix
 * uplo = 1 if upper triangular part of a is supplied, 0 if lower
 * unit = 1 if a is unit triangular, 0 if non-unit
 */
double dtrmv(bool trans,
             bool uplo,
             bool unit,
             size_t m,
             size_t n,
             int kl,
             int ku,
             double alpha,
             double beta,
             double **a,
             double *x,
             double *y,
             double *at)
{
	(void)m;
	(void)kl;
	(void)ku;
	(void)alpha;
	(void)beta;
	(void)*y;
	(void)*at;

	if (n == 0)
		return 1;

	size_t i, j;
	double temp;

	if (!trans) {
		/* x = a * x */
		if (uplo) {
			/* Upper triangular */
			for (j = 0; j < n; j++) {
				if (x[j] != 0) {
					temp = x[j];
					for (i = 0; i < j - 1; i++)
						x[i] += temp * a[i][j];
					if (!unit)
						x[j] = x[j] * a[j][j];
				}
			}
		} else {
			/* Lower triangular */
			for (j = n - 1; j > -1; j--) {
				if (x[j] != 0) {
					temp = x[j];
					for (i = n - 1; i > j; i--)
						x[i] += temp * a[i][j];
					if (!unit)
						x[j] = x[j] * a[j][j];
				}
			}
		}
	} else {
		/* x = a^T * x */
		if (uplo) {
			/* Upper trianglar */
			for (j = n - 1; j > -1; j--) {
				temp = x[j];
				if (!unit)
					temp = temp * a[j][j];
				for (i = j - 1; i > -1; i--)
					temp += a[i][j] * x[i];
				x[j] = temp;
			}
		} else {
			/* Lower triangular */
			for (j = 0; j < n; j++) {
				temp = x[j];
				if (!unit)
					temp = temp * a[j][j];
				for (i = j + 1; i < n; i++)
					temp += a[i][j] * x[i];
				x[j] = temp;
			}
		}
	}
	return 1;
}
