#pragma once

#include <array>

// HitReg is a compile‑time configurable hit record:
// the tag list decides which fields exist, and any write to a
// field that isn’t present is simply ignored.

#if defined(__CUDACC__) || defined(__HIPCC__)
#define RT_HD __host__ __device__
#else
#define RT_HD
#endif

namespace portableRT {
struct Empty {

	template <class T> RT_HD constexpr Empty &operator=(T &&) noexcept {
		return *this;
	} // We ignore assignments.
	template <class T>
	RT_HD constexpr operator T() const = delete; // Avoid compilation on data reads.
};

#define ALL_TAGS                                                                                   \
	portableRT::filter::uv, portableRT::filter::t, portableRT::filter::primitive_id,               \
	    portableRT::filter::p, portableRT::filter::valid

template <bool HasUV, bool HasT, bool HasPrimitiveId, bool HasP, bool HasValid> struct HitRegImpl {
	using has_uv = std::bool_constant<HasUV>;
	using has_t = std::bool_constant<HasT>;
	using has_primitive_id = std::bool_constant<HasPrimitiveId>;
	using has_p = std::bool_constant<HasP>;
	using has_valid = std::bool_constant<HasValid>;

	std::conditional_t<HasUV, float, Empty> u;
	std::conditional_t<HasUV, float, Empty> v;
	std::conditional_t<HasT, float, Empty> t;
	std::conditional_t<HasPrimitiveId, uint32_t, Empty> primitive_id;
	std::conditional_t<HasValid, bool, Empty> valid;
	std::conditional_t<HasP, float, Empty> px;
	std::conditional_t<HasP, float, Empty> py;
	std::conditional_t<HasP, float, Empty> pz;
};

template <typename T, typename... List> constexpr bool has_tag = (std::is_same_v<T, List> || ...);

namespace filter {
struct uv {};
struct t {};
struct primitive_id {};
struct p {};
struct valid {};
} // namespace filter

template <class... Tags>
using HitReg = HitRegImpl<has_tag<filter::uv, Tags...>, has_tag<filter::t, Tags...>,
                          has_tag<filter::primitive_id, Tags...>, has_tag<filter::p, Tags...>,
                          has_tag<filter::valid, Tags...>>;

using FullHitReg = HitReg<ALL_TAGS>;

template <class... Tags> HitReg<Tags...> slice(const FullHitReg &hit) {
	HitReg<Tags...> res;
	if constexpr (has_tag<filter::uv, Tags...>) {
		res.u = hit.u;
		res.v = hit.v;
	}
	if constexpr (has_tag<filter::t, Tags...>) {
		res.t = hit.t;
	}
	if constexpr (has_tag<filter::primitive_id, Tags...>) {
		res.primitive_id = hit.primitive_id;
	}
	if constexpr (has_tag<filter::p, Tags...>) {
		res.px = hit.px;
		res.py = hit.py;
		res.pz = hit.pz;
	}
	if constexpr (has_tag<filter::valid, Tags...>) {
		res.valid = hit.valid;
	}
	return res;
}

#define CAT2(a, b) a##b
#define CAT(a, b) CAT2(a, b)

/* —— contador 0‑6 —— */
#define NARGS_IMPL(_1, _2, _3, _4, _5, _6, _7, ...) _7
#define NARGS(...) NARGS_IMPL(__VA_ARGS__, 6, 5, 4, 3, 2, 1, 0)

/* —— unir nombres —— */
#define JOIN0() /* 0 tags: vacío  */
#define JOIN1(a) a
#define JOIN2(a, b) a##_##b
#define JOIN3(a, b, c) a##_##b##_##c
#define JOIN4(a, b, c, d) a##_##b##_##c##_##d
#define JOIN5(a, b, c, d, e) a##_##b##_##c##_##d##_##e
#define JOIN6(a, b, c, d, e, f) a##_##b##_##c##_##d##_##e##_##f

#define JOIN_DISPATCH(n, ...) CAT(JOIN, n)(__VA_ARGS__)
#define JOIN(...) JOIN_DISPATCH(NARGS(__VA_ARGS__), __VA_ARGS__)

/* —— prefijar filtros —— */
#define FILTER(x) portableRT::filter::x
#define ADD_FILTER0() /* sin filtros   */
#define ADD_FILTER1(a) FILTER(a)
#define ADD_FILTER2(a, b) FILTER(a), FILTER(b)
#define ADD_FILTER3(a, b, c) FILTER(a), FILTER(b), FILTER(c)
#define ADD_FILTER4(a, b, c, d) FILTER(a), FILTER(b), FILTER(c), FILTER(d)
#define ADD_FILTER5(a, b, c, d, e) FILTER(a), FILTER(b), FILTER(c), FILTER(d), FILTER(e)
#define ADD_FILTER6(a, b, c, d, e, f)                                                              \
	FILTER(a), FILTER(b), FILTER(c), FILTER(d), FILTER(e), FILTER(f)

#define ADD_DISPATCH(n, ...) CAT(ADD_FILTER, n)(__VA_ARGS__)
#define ADD_FILTER(...) ADD_DISPATCH(NARGS(__VA_ARGS__), __VA_ARGS__)

/* —— nombres de símbolos —— */
#define PARAMS_NAME(...) CAT(g_params_, JOIN(__VA_ARGS__))
#define KERNEL_NAME(...) CAT(__raygen__rg__, JOIN(__VA_ARGS__))

/* —— macro de usuario —— */
#define DEFINE_RAYGEN(...)                                                                         \
	extern "C" __constant__ Params<ADD_FILTER(__VA_ARGS__)> PARAMS_NAME(__VA_ARGS__);              \
	extern "C" __global__ void KERNEL_NAME(__VA_ARGS__)() { raygen_body(PARAMS_NAME(__VA_ARGS__)); }

#define TAG_COMBOS                                                                                 \
	X(uv)                                                                                          \
	X(t)                                                                                           \
	X(primitive_id)                                                                                \
	X(p)                                                                                           \
	X(valid)                                                                                       \
	X(uv, t)                                                                                       \
	X(uv, primitive_id)                                                                            \
	X(uv, p)                                                                                       \
	X(uv, valid)                                                                                   \
	X(t, primitive_id)                                                                             \
	X(t, p)                                                                                        \
	X(t, valid)                                                                                    \
	X(primitive_id, p)                                                                             \
	X(primitive_id, valid)                                                                         \
	X(p, valid)                                                                                    \
	X(uv, t, primitive_id)                                                                         \
	X(uv, t, p)                                                                                    \
	X(uv, t, valid)                                                                                \
	X(uv, primitive_id, p)                                                                         \
	X(uv, primitive_id, valid)                                                                     \
	X(uv, p, valid)                                                                                \
	X(t, primitive_id, p)                                                                          \
	X(t, primitive_id, valid)                                                                      \
	X(t, p, valid)                                                                                 \
	X(primitive_id, p, valid)                                                                      \
	X(uv, t, primitive_id, p)                                                                      \
	X(uv, t, primitive_id, valid)                                                                  \
	X(uv, t, p, valid)                                                                             \
	X(uv, primitive_id, p, valid)                                                                  \
	X(t, primitive_id, p, valid)                                                                   \
	X(uv, t, primitive_id, p, valid)

} // namespace portableRT