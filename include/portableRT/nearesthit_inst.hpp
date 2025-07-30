// For backends that require precompilation of all instantiations (those that use a custom
// compiler).
#define INSTANTIATE_HITREG(BACKEND, ...)                                                           \
	template std::vector<portableRT::HitReg<ADD_FILTER(__VA_ARGS__)>>                              \
	BACKEND::nearest_hits<ADD_FILTER(__VA_ARGS__)>(const std::vector<portableRT::Ray> &);