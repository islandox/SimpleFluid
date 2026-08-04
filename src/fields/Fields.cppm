module;

#if !defined(SIMPLEFLUID_USE_STD_MODULE)
#include "cmake/StandardHeaders.hh"
#endif

#include "fields/BoundaryFaceField.hh"
#include "fields/CellField.hh"
#include "fields/FaceField.hh"
#include "fields/Field.hh"
#include "fields/FieldStored.hh"
#include "fields/TensorCellField.hh"
#include "fields/VectorCellField.hh"
#include "fields/VectorFaceField.hh"

export module SimpleFluid.Fields;

#if defined(SIMPLEFLUID_USE_STD_MODULE)
import std;
#endif

export import SimpleFluid.Mesh;

export namespace SimpleFluid
{
using ::SimpleFluid::AnyFieldStored;
using ::SimpleFluid::BoundaryFaceField;
using ::SimpleFluid::BoundaryFaceLocation;
using ::SimpleFluid::CellField;
using ::SimpleFluid::CellFieldBase;
using ::SimpleFluid::CellLocation;
using ::SimpleFluid::FaceField;
using ::SimpleFluid::FaceFieldBase;
using ::SimpleFluid::FaceLocation;
using ::SimpleFluid::Field;
using ::SimpleFluid::FieldStored;
using ::SimpleFluid::ScalarBoundaryFaceFieldDescriptor;
using ::SimpleFluid::ScalarBoundaryFaceFieldStored;
using ::SimpleFluid::ScalarCellFieldDescriptor;
using ::SimpleFluid::ScalarCellFieldStored;
using ::SimpleFluid::ScalarFaceFieldDescriptor;
using ::SimpleFluid::ScalarFaceFieldStored;
using ::SimpleFluid::TensorCellField;
using ::SimpleFluid::VectorBoundaryFaceFieldDescriptor;
using ::SimpleFluid::VectorBoundaryFaceFieldStored;
using ::SimpleFluid::VectorCellField;
using ::SimpleFluid::VectorCellFieldDescriptor;
using ::SimpleFluid::VectorCellFieldStored;
using ::SimpleFluid::VectorFaceField;
using ::SimpleFluid::VectorFaceFieldDescriptor;
using ::SimpleFluid::VectorFaceFieldStored;
}
