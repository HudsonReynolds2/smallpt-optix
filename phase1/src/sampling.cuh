#pragma once

//-----------------------------------------------------------------------------
// Includes
//-----------------------------------------------------------------------------
#pragma region

#include "vector.hpp"

#pragma endregion

//-----------------------------------------------------------------------------
// Declarations and Definitions
//-----------------------------------------------------------------------------
namespace smallpt {

	__device__ inline Vector3 UniformSampleOnHemisphere(float u1,
	                                                    float u2) {
		// u1 := cos_theta
		const float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - u1 * u1));
		const float phi = 2.0f * g_pi * u2;
		return {
			cosf(phi) * sin_theta,
			sinf(phi) * sin_theta,
			u1
		};
	}

	__device__ inline Vector3 CosineWeightedSampleOnHemisphere(float u1,
	                                                           float u2) {
		const float cos_theta = sqrtf(1.0f - u1);
		const float sin_theta = sqrtf(u1);
		const float phi = 2.0f * g_pi * u2;
		return {
			cosf(phi) * sin_theta,
			sinf(phi) * sin_theta,
			cos_theta
		};
	}
}
