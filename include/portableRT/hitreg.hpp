#pragma once

// HitReg is a compile‑time configurable hit record:
// the tag list decides which fields exist, and any write to a
// field that isn’t present is simply ignored.

namespace portableRT {
struct Empty {
	template <class T> constexpr Empty &operator=(T &&) noexcept {
		return *this;
	} // We ignore assignments.
	template <class T> constexpr operator T() const = delete; // Avoid compilation on data reads.
};

#define ALL_TAGS filter::uv, filter::t, filter::primitive_id, filter::p, filter::valid

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
	std::conditional_t<HasP, std::array<float, 3>, Empty> p;
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
		res.p = hit.p;
	}
	if constexpr (has_tag<filter::valid, Tags...>) {
		res.valid = hit.valid;
	}
	return res;
}
} // namespace portableRT