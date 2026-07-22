module;

#include "cmake/StandardHeaders.hh"

#include "dataclass/typedefs.hh"
#include "dataclass/vec3.hh"
#include "dataclass/DBNode.hh"
#include "dataclass/Database.hh"
#include "dataclass/RandomAccessView.hh"
#include "dataclass/TpetraTypes.hh"

export module SimpleFluid.Core;

export namespace SimpleFluid
{
using ::SimpleFluid::real_t;
using ::SimpleFluid::global_index_t;
using ::SimpleFluid::local_index_t;
using ::SimpleFluid::ArrReal;
using ::SimpleFluid::ArrInt;
using ::SimpleFluid::ArrString;
using ::SimpleFluid::ArrBool;
using ::SimpleFluid::Arr;
using ::SimpleFluid::Vec3DReal;
using ::SimpleFluid::Vec3D;
using ::SimpleFluid::UP;
using ::SimpleFluid::SP;
using ::SimpleFluid::WP;
using ::SimpleFluid::Dimension;
using ::SimpleFluid::X;
using ::SimpleFluid::Y;
using ::SimpleFluid::Z;
using ::SimpleFluid::vec3;
using ::SimpleFluid::DBNode;
using ::SimpleFluid::Database;
using ::SimpleFluid::RandomAccessView;
using ::SimpleFluid::TpetraTypePack;
using ::SimpleFluid::TpetraTypes;
using ::SimpleFluid::DefaultTpetraTypes;
}

// Keep the dependencies of exported CrsMatrix specializations reachable when
// Clang instantiates default Tpetra types through a project module boundary.
export namespace Tpetra::MMdetails
{
using ::Tpetra::MMdetails::KernelWrappers;
using ::Tpetra::MMdetails::KernelWrappers2;
}
