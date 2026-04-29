#pragma once

//-----------------------------------------------------------------------------
// Includes
//-----------------------------------------------------------------------------
#pragma region

#include "math.hpp"

#pragma endregion

//-----------------------------------------------------------------------------
// Declarations and Definitions
//-----------------------------------------------------------------------------
namespace smallpt {

	//-------------------------------------------------------------------------
	// Vector3
	//-------------------------------------------------------------------------

	struct Vector3 {

	public:

		//---------------------------------------------------------------------
		// Constructors and Destructors
		//---------------------------------------------------------------------

		__host__ __device__ explicit Vector3(float xyz = 0.0f) noexcept
			: Vector3(xyz, xyz, xyz) {}
		__host__ __device__ Vector3(float x, float y, float z) noexcept
			: m_x(x), m_y(y), m_z(z) {}
		Vector3(const Vector3& v) noexcept = default;
		Vector3(Vector3&& v) noexcept = default;
		~Vector3() = default;

		//---------------------------------------------------------------------
		// Assignment Operators
		//---------------------------------------------------------------------

		Vector3& operator=(const Vector3& v) = default;
		Vector3& operator=(Vector3&& v) = default;

		//---------------------------------------------------------------------
		// Member Methods
		//---------------------------------------------------------------------

		__device__ bool HasNaNs() const {
			return isnan(m_x) || isnan(m_y) || isnan(m_z);
		}

		__device__ const Vector3 operator-() const {
			return { -m_x, -m_y, -m_z };
		}

		__device__ const Vector3 operator+(const Vector3& v) const {
			return { m_x + v.m_x, m_y + v.m_y, m_z + v.m_z };
		}
		__device__ const Vector3 operator-(const Vector3& v) const {
			return { m_x - v.m_x, m_y - v.m_y, m_z - v.m_z };
		}
		__device__ const Vector3 operator*(const Vector3& v) const {
			return { m_x * v.m_x, m_y * v.m_y, m_z * v.m_z };
		}
		__device__ const Vector3 operator/(const Vector3& v) const {
			return { m_x / v.m_x, m_y / v.m_y, m_z / v.m_z };
		}
		__device__ const Vector3 operator+(float a) const {
			return { m_x + a, m_y + a, m_z + a };
		}
		__device__ const Vector3 operator-(float a) const {
			return { m_x - a, m_y - a, m_z - a };
		}
		__device__ const Vector3 operator*(float a) const {
			return { m_x * a, m_y * a, m_z * a };
		}
		__device__ const Vector3 operator/(float a) const {
			const float inv_a = 1.0f / a;
			return { m_x * inv_a, m_y * inv_a, m_z * inv_a };
		}

		__device__ Vector3& operator+=(const Vector3& v) {
			m_x += v.m_x;
			m_y += v.m_y;
			m_z += v.m_z;
			return *this;
		}
		__device__ Vector3& operator-=(const Vector3& v) {
			m_x -= v.m_x;
			m_y -= v.m_y;
			m_z -= v.m_z;
			return *this;
		}
		__device__ Vector3& operator*=(const Vector3& v) {
			m_x *= v.m_x;
			m_y *= v.m_y;
			m_z *= v.m_z;
			return *this;
		}
		__device__ Vector3& operator/=(const Vector3& v) {
			m_x /= v.m_x;
			m_y /= v.m_y;
			m_z /= v.m_z;
			return *this;
		}
		__device__ Vector3& operator+=(float a) {
			m_x += a;
			m_y += a;
			m_z += a;
			return *this;
		}
		__device__ Vector3& operator-=(float a) {
			m_x -= a;
			m_y -= a;
			m_z -= a;
			return *this;
		}
		__device__ Vector3& operator*=(float a) {
			m_x *= a;
			m_y *= a;
			m_z *= a;
			return *this;
		}
		__device__ Vector3& operator/=(float a) {
			const float inv_a = 1.0f / a;
			m_x *= inv_a;
			m_y *= inv_a;
			m_z *= inv_a;
			return *this;
		}

		__device__ float Dot(const Vector3& v) const {
			return m_x * v.m_x + m_y * v.m_y + m_z * v.m_z;
		}
		__device__ const Vector3 Cross(const Vector3& v) const {
			return {
				m_y * v.m_z - m_z * v.m_y,
				m_z * v.m_x - m_x * v.m_z,
				m_x * v.m_y - m_y * v.m_x
			};
		}

		__device__ bool operator==(const Vector3& rhs) const {
			return m_x == rhs.m_x && m_y == rhs.m_y && m_z == rhs.m_z;
		}
		__device__ bool operator!=(const Vector3& rhs) const {
			return !(*this == rhs);
		}

		__device__ float& operator[](std::size_t i) {
			return (&m_x)[i];
		}
		__device__ float operator[](std::size_t i) const {
			return (&m_x)[i];
		}

		__device__ std::size_t MinDimension() const {
			return (m_x < m_y && m_x < m_z) ? 0u : ((m_y < m_z) ? 1u : 2u);
		}
		__device__ std::size_t MaxDimension() const {
			return (m_x > m_y && m_x > m_z) ? 0u : ((m_y > m_z) ? 1u : 2u);
		}
		__device__ float Min() const {
			return fminf(m_x, fminf(m_y, m_z));
		}
		__device__ float Max() const {
			return fmaxf(m_x, fmaxf(m_y, m_z));
		}

		__device__ float Norm2_squared() const {
			return m_x * m_x + m_y * m_y + m_z * m_z;
		}

		__device__ float Norm2() const {
			return sqrtf(Norm2_squared());
		}

		__device__ void Normalize() {
			const float a = 1.0f / Norm2();
			m_x *= a;
			m_y *= a;
			m_z *= a;
		}

		//---------------------------------------------------------------------
		// Member Variables
		//---------------------------------------------------------------------

		float m_x, m_y, m_z;
	};

	//-------------------------------------------------------------------------
	// Vector3 Utilities
	//-------------------------------------------------------------------------

	__device__ inline const Vector3 operator+(float a, const Vector3& v) {
		return { a + v.m_x, a + v.m_y, a + v.m_z };
	}

	__device__ inline const Vector3 operator-(float a, const Vector3& v) {
		return { a - v.m_x, a - v.m_y, a - v.m_z };
	}

	__device__ inline const Vector3 operator*(float a, const Vector3& v) {
		return { a * v.m_x, a * v.m_y, a * v.m_z };
	}

	__device__ inline const Vector3 operator/(float a, const Vector3& v) {
		return { a / v.m_x, a / v.m_y, a / v.m_z };
	}

	__device__ inline const Vector3 Sqrt(const Vector3& v) {
		return {
			sqrtf(v.m_x),
			sqrtf(v.m_y),
			sqrtf(v.m_z)
		};
	}

	__device__ inline const Vector3 Pow(const Vector3& v, float a) {
		return {
			powf(v.m_x, a),
			powf(v.m_y, a),
			powf(v.m_z, a)
		};
	}

	__device__ inline const Vector3 Abs(const Vector3& v) {
		return {
			fabsf(v.m_x),
			fabsf(v.m_y),
			fabsf(v.m_z)
		};
	}

	__device__ inline const Vector3 Min(const Vector3& v1, const Vector3& v2) {
		return {
			fminf(v1.m_x, v2.m_x),
			fminf(v1.m_y, v2.m_y),
			fminf(v1.m_z, v2.m_z)
		};
	}

	__device__ inline const Vector3 Max(const Vector3& v1, const Vector3& v2) {
		return {
			fmaxf(v1.m_x, v2.m_x),
			fmaxf(v1.m_y, v2.m_y),
			fmaxf(v1.m_z, v2.m_z)
		};
	}

	__device__ inline const Vector3 Round(const Vector3& v) {
		return {
			roundf(v.m_x),
			roundf(v.m_y),
			roundf(v.m_z)
		};
	}

	__device__ inline const Vector3 Floor(const Vector3& v) {
		return {
			floorf(v.m_x),
			floorf(v.m_y),
			floorf(v.m_z)
		};
	}

	__device__ inline const Vector3 Ceil(const Vector3& v) {
		return {
			ceilf(v.m_x),
			ceilf(v.m_y),
			ceilf(v.m_z)
		};
	}

	__device__ inline const Vector3 Trunc(const Vector3& v) {
		return {
			truncf(v.m_x),
			truncf(v.m_y),
			truncf(v.m_z)
		};
	}

	__device__ inline const Vector3 Clamp(const Vector3& v,
	                                      float low  = 0.0f,
	                                      float high = 1.0f) {
		return {
			Clamp(v.m_x, low, high),
			Clamp(v.m_y, low, high),
			Clamp(v.m_z, low, high) }
		;
	}

	__device__ inline const Vector3 Lerp(float a,
	                                     const Vector3& v1,
	                                     const Vector3& v2) {
		return v1 + a * (v2 - v1);
	}

	template< std::size_t X, std::size_t Y, std::size_t Z >
	__device__ inline const Vector3 Permute(const Vector3& v) {
		return { v[X], v[Y], v[Z] };
	}

	__device__ inline const Vector3 Normalize(const Vector3& v) {
		const float a = 1.0f / v.Norm2();
		return a * v;
	}
}
