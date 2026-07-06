module;

#include "fields/BoundaryFaceField.hh"
#include "fields/CellField.hh"
#include "fields/FaceField.hh"
#include "fields/Field.hh"
#include "fields/FieldStored.hh"
#include "fields/TensorCellField.hh"
#include "fields/VectorCellField.hh"
#include "fields/VectorFaceField.hh"

export module SimpleFluid.Fields;

export namespace SimpleFluid
{
using ::SimpleFluid::AnyFieldStored;
using ::SimpleFluid::Arr;
using ::SimpleFluid::BoundaryFaceField;
using ::SimpleFluid::BoundaryFaceLocation;
using ::SimpleFluid::CellField;
using ::SimpleFluid::CellFieldBase;
using ::SimpleFluid::CellLocation;
using ::SimpleFluid::DefaultTpetraTypes;
using ::SimpleFluid::FaceField;
using ::SimpleFluid::FaceFieldBase;
using ::SimpleFluid::FaceLocation;
using ::SimpleFluid::Field;
using ::SimpleFluid::FieldStored;
using ::SimpleFluid::Mesh;
using ::SimpleFluid::ScalarBoundaryFaceFieldDescriptor;
using ::SimpleFluid::ScalarBoundaryFaceFieldStored;
using ::SimpleFluid::ScalarCellFieldDescriptor;
using ::SimpleFluid::ScalarCellFieldStored;
using ::SimpleFluid::ScalarFaceFieldDescriptor;
using ::SimpleFluid::ScalarFaceFieldStored;
using ::SimpleFluid::SP;
using ::SimpleFluid::TensorCellField;
using ::SimpleFluid::TpetraTypePack;
using ::SimpleFluid::TpetraTypes;
using ::SimpleFluid::VectorBoundaryFaceFieldDescriptor;
using ::SimpleFluid::VectorBoundaryFaceFieldStored;
using ::SimpleFluid::VectorCellField;
using ::SimpleFluid::VectorCellFieldDescriptor;
using ::SimpleFluid::VectorCellFieldStored;
using ::SimpleFluid::VectorFaceField;
using ::SimpleFluid::VectorFaceFieldDescriptor;
using ::SimpleFluid::VectorFaceFieldStored;
using ::SimpleFluid::real_t;
using ::SimpleFluid::vec3;
}
