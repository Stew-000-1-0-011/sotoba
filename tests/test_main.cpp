#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "sotoba/math/vec.hpp"
#include "sotoba/math/quaternion.hpp"
#include "sotoba/math/se3.hpp"

#include "sotoba/surface/rectangle.hpp"
#include "sotoba/surface/cylinder.hpp"
#include "sotoba/surface/box.hpp"

#include "sotoba/icp_resource/svd_icp.hpp"
#include "sotoba/icp_resource/normal_known_icp.hpp"