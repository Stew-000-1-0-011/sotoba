import subprocess

def format_and_read(filepath: str, content: str) -> str:
	subprocess.run(["clang-format-20", "-i", filepath])
	with open(filepath, "r") as f:
		s = f.read()
		content += '\n'.join(l for l in s.splitlines() if not l.strip().startswith("#include")) + '\n\n'
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

#ifdef sotoba_ENABLE_TESTING
#include <doctest.h>
#endif

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
content += "\n\nint main(){}\n"

with open("src/bin/expanded_cpp.cpp", "w") as f:
	f.write(content)

subprocess.run(["clang-format-20", "-i", "src/bin/expanded_cpp.cpp"])