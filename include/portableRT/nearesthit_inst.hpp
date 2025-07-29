// For backends that require precompilation of all instantiations (those that use a custom
// compiler).
#define NEAREST_HITS_INSTANTIATE(BACKEND)                                                          \
	template std::vector<HitReg<>> BACKEND::nearest_hits<>(const std::vector<Ray> &);              \
	template std::vector<HitReg<filter::uv>> BACKEND::nearest_hits<filter::uv>(                    \
	    const std::vector<Ray> &);                                                                 \
	template std::vector<HitReg<filter::t>> BACKEND::nearest_hits<filter::t>(                      \
	    const std::vector<Ray> &);                                                                 \
	template std::vector<HitReg<filter::primitive_id>>                                             \
	BACKEND::nearest_hits<filter::primitive_id>(const std::vector<Ray> &);                         \
	template std::vector<HitReg<filter::p>> BACKEND::nearest_hits<filter::p>(                      \
	    const std::vector<Ray> &);                                                                 \
	template std::vector<HitReg<filter::valid>> BACKEND::nearest_hits<filter::valid>(              \
	    const std::vector<Ray> &);                                                                 \
	template std::vector<HitReg<filter::uv, filter::t>>                                            \
	BACKEND::nearest_hits<filter::uv, filter::t>(const std::vector<Ray> &);                        \
	template std::vector<HitReg<filter::uv, filter::primitive_id>>                                 \
	BACKEND::nearest_hits<filter::uv, filter::primitive_id>(const std::vector<Ray> &);             \
	template std::vector<HitReg<filter::uv, filter::p>>                                            \
	BACKEND::nearest_hits<filter::uv, filter::p>(const std::vector<Ray> &);                        \
	template std::vector<HitReg<filter::uv, filter::valid>>                                        \
	BACKEND::nearest_hits<filter::uv, filter::valid>(const std::vector<Ray> &);                    \
	template std::vector<HitReg<filter::t, filter::primitive_id>>                                  \
	BACKEND::nearest_hits<filter::t, filter::primitive_id>(const std::vector<Ray> &);              \
	template std::vector<HitReg<filter::t, filter::p>>                                             \
	BACKEND::nearest_hits<filter::t, filter::p>(const std::vector<Ray> &);                         \
	template std::vector<HitReg<filter::t, filter::valid>>                                         \
	BACKEND::nearest_hits<filter::t, filter::valid>(const std::vector<Ray> &);                     \
	template std::vector<HitReg<filter::primitive_id, filter::p>>                                  \
	BACKEND::nearest_hits<filter::primitive_id, filter::p>(const std::vector<Ray> &);              \
	template std::vector<HitReg<filter::primitive_id, filter::valid>>                              \
	BACKEND::nearest_hits<filter::primitive_id, filter::valid>(const std::vector<Ray> &);          \
	template std::vector<HitReg<filter::p, filter::valid>>                                         \
	BACKEND::nearest_hits<filter::p, filter::valid>(const std::vector<Ray> &);                     \
	template std::vector<HitReg<filter::uv, filter::t, filter::primitive_id>>                      \
	BACKEND::nearest_hits<filter::uv, filter::t, filter::primitive_id>(const std::vector<Ray> &);  \
	template std::vector<HitReg<filter::uv, filter::t, filter::p>>                                 \
	BACKEND::nearest_hits<filter::uv, filter::t, filter::p>(const std::vector<Ray> &);             \
	template std::vector<HitReg<filter::uv, filter::t, filter::valid>>                             \
	BACKEND::nearest_hits<filter::uv, filter::t, filter::valid>(const std::vector<Ray> &);         \
	template std::vector<HitReg<filter::uv, filter::primitive_id, filter::p>>                      \
	BACKEND::nearest_hits<filter::uv, filter::primitive_id, filter::p>(const std::vector<Ray> &);  \
	template std::vector<HitReg<filter::uv, filter::primitive_id, filter::valid>>                  \
	BACKEND::nearest_hits<filter::uv, filter::primitive_id, filter::valid>(                        \
	    const std::vector<Ray> &);                                                                 \
	template std::vector<HitReg<filter::uv, filter::p, filter::valid>>                             \
	BACKEND::nearest_hits<filter::uv, filter::p, filter::valid>(const std::vector<Ray> &);         \
	template std::vector<HitReg<filter::t, filter::primitive_id, filter::p>>                       \
	BACKEND::nearest_hits<filter::t, filter::primitive_id, filter::p>(const std::vector<Ray> &);   \
	template std::vector<HitReg<filter::t, filter::primitive_id, filter::valid>>                   \
	BACKEND::nearest_hits<filter::t, filter::primitive_id, filter::valid>(                         \
	    const std::vector<Ray> &);                                                                 \
	template std::vector<HitReg<filter::t, filter::p, filter::valid>>                              \
	BACKEND::nearest_hits<filter::t, filter::p, filter::valid>(const std::vector<Ray> &);          \
	template std::vector<HitReg<filter::primitive_id, filter::p, filter::valid>>                   \
	BACKEND::nearest_hits<filter::primitive_id, filter::p, filter::valid>(                         \
	    const std::vector<Ray> &);                                                                 \
	template std::vector<HitReg<filter::uv, filter::t, filter::primitive_id, filter::p>>           \
	BACKEND::nearest_hits<filter::uv, filter::t, filter::primitive_id, filter::p>(                 \
	    const std::vector<Ray> &);                                                                 \
	template std::vector<HitReg<filter::uv, filter::t, filter::primitive_id, filter::valid>>       \
	BACKEND::nearest_hits<filter::uv, filter::t, filter::primitive_id, filter::valid>(             \
	    const std::vector<Ray> &);                                                                 \
	template std::vector<HitReg<filter::uv, filter::t, filter::p, filter::valid>>                  \
	BACKEND::nearest_hits<filter::uv, filter::t, filter::p, filter::valid>(                        \
	    const std::vector<Ray> &);                                                                 \
	template std::vector<HitReg<filter::uv, filter::primitive_id, filter::p, filter::valid>>       \
	BACKEND::nearest_hits<filter::uv, filter::primitive_id, filter::p, filter::valid>(             \
	    const std::vector<Ray> &);                                                                 \
	template std::vector<HitReg<filter::t, filter::primitive_id, filter::p, filter::valid>>        \
	BACKEND::nearest_hits<filter::t, filter::primitive_id, filter::p, filter::valid>(              \
	    const std::vector<Ray> &);                                                                 \
	template std::vector<                                                                          \
	    HitReg<filter::uv, filter::t, filter::primitive_id, filter::p, filter::valid>>             \
	BACKEND::nearest_hits<filter::uv, filter::t, filter::primitive_id, filter::p, filter::valid>(  \
	    const std::vector<Ray> &);
