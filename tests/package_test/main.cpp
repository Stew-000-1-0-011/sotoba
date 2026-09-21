// find_package(sotoba) 経由での利用を検証するための最小の実行ファイル。
// sotoba/icp_resource/normal_known_icp.hpp を include し、
// NormalKnownResource を1つ構築するだけ (ヘッダが通ること、および
// Eigen3 の伝播が効いていることを確認する)。

#include <array>
#include <tuple>
#include <vector>

#include "sotoba/icp_resource/normal_known_icp.hpp"
#include "sotoba/surf_obj_id.hpp"
#include "sotoba/surface/rectangle.hpp"

auto main() -> int {
	using sotoba::ObjSurfId;
	using sotoba::icp_resource::NormalKnownResource;
	using sotoba::surface::Rectangle;

	std::tuple<std::vector<Rectangle>> surfs{};
	std::array<std::vector<ObjSurfId>, 1> osids{};

	auto icp = NormalKnownResource<Rectangle>{std::move(surfs), std::move(osids), 0, 0};

	return icp.points_capacity() == 0 ? 0 : 1;
}
