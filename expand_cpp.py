import subprocess

def format_and_read(filepath: str, content: str) -> str:
	subprocess.run(["clang-format-20", "-i", filepath])
	with open(filepath, "r") as f:
		s = f.read()
		content += '\n'.join(l for l in s.splitlines() if not l.strip().startswith("#") or l.strip().startswith("#define")) + '\n\n'
	return content

content: str = '''
#include <concepts>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>

#include "doctest.h"

#define SYCL_SIMPLE_SWIZZLES
#include <sycl/sycl.hpp>

'''

content = format_and_read("include/sotoba/exception.hpp", content)
content = format_and_read("include/sotoba/fundamental.hpp", content)
content = format_and_read("include/sotoba/philox.hpp", content)
content = format_and_read("include/sotoba/surface.hpp", content)
content = format_and_read("include/sotoba/rectangle.hpp", content)
content = format_and_read("include/sotoba/cylinder.hpp", content)
content = format_and_read("include/sotoba/lidar.hpp", content)
content = format_and_read("src/lib.cpp", content)

with open("expanded_cpp.cpp", "w") as f:
	f.write(content)

subprocess.run(["clang-format-20", "-i", "expanded_cpp.cpp"])