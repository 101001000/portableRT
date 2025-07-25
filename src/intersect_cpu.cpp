#include "../include/portableRT/intersect_cpu.hpp"
#include <array>
#include <fstream>
#include <functional>
#include <thread>

namespace portableRT {

bool CPUBackend::is_available() const { return true; }
void CPUBackend::init() {}
void CPUBackend::shutdown() {}
void CPUBackend::set_tris(const Tris &tris) { m_bvh.build(tris); }

std::string CPUBackend::device_name() const {
	std::ifstream cpuinfo("/proc/cpuinfo");
	std::string line;
	while (std::getline(cpuinfo, line)) {
		if (line.rfind("model name", 0) == 0) {
			auto pos = line.find(':');
			if (pos != std::string::npos) {
				size_t start = line.find_first_not_of(" \t", pos + 1);
				return line.substr(start);
			}
		}
	}
	return "unsupported";
}

} // namespace portableRT
