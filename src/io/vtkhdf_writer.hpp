#pragma once

// Writer for OpenSees' VTKHDF recorder layout (SRC/recorder/VTKHDF_Recorder.cpp, OpenSees
// commit 955ceb0), so that femora.results and ParaView read FemoraX output unchanged.
//
// The file is a VTK HDF "UnstructuredGrid" (Version 2.3) with a static mesh and one entry per
// recorded step in /VTKHDF/Steps:
//
//   /VTKHDF                 attrs Version = [2, 3] (int32), Type = "UnstructuredGrid"
//     Points [n, 3] f64, Connectivity [c] i32, Offsets [m + 1] i32, Types [m] u8
//     NumberOfPoints, NumberOfCells, NumberOfConnectivityIds [steps] i32 (one per step)
//     PointData/  nodeTags [n] i32, displacement / velocity / acceleration [steps * n, 3] f64
//     CellData/   eleTags [m] i32, stress3D6 / strain3D6 [steps * m, 6] f64 (when recorded)
//     Steps/      attr NSteps; Values [steps] f64 (the times); NumberOfParts, PartOffsets,
//                 PointOffsets [steps] i32; CellOffsets, ConnectivityIdOffsets [steps, 1] i32;
//                 PointDataOffsets/<field> [steps] i32 (row of each step's block; nodeTags: 0);
//                 CellDataOffsets/eleTags [steps] i32 (0), CellDataOffsets/<field> [steps]
//                 i32 (row of each step's block); FieldDataOffsets/, FieldDataSizes/
//
// Datasets are created empty with the same element types, chunk shapes and unlimited first
// dimension as OpenSees, and grow as OpenSees grows them. With single precision (a FemoraX
// option, not in OpenSees) the response datasets (PointData displacement / velocity /
// acceleration, CellData stress3D6 / strain3D6) are float32, each value the FP64 result
// rounded to nearest by HDF5's conversion; the mesh, the times and everything else stay as
// above. Computation is FP64 either way. The mesh is written by
// write_mesh() (OpenSees: on the recorder's first record() call), each recorded step by
// write_step(). Host code; not thread-safe: one thread owns a writer.

#include <cstdint>
#include <string>
#include <vector>

namespace femorax::io {

struct VtkhdfMesh {
  std::vector<double> points;               // [n * 3]
  std::vector<std::int32_t> connectivity;   // point indices, element by element
  std::vector<std::int32_t> offsets;        // [m + 1], cumulative, starting at 0
  std::vector<std::uint8_t> types;          // [m], VTK cell types
  std::vector<std::int32_t> node_tags;      // [n]
  std::vector<std::int32_t> ele_tags;       // [m]
};

// VTK cell type of an 8-node hexahedron (VTK_HEXAHEDRON).
inline constexpr std::uint8_t kVtkHexahedron = 12;

class VtkhdfWriter {
 public:
  // Create (truncate) `file` with the empty layout. `fields` are the PointData names, in
  // the order write_step() receives them: any of "displacement", "velocity",
  // "acceleration"; `cell_fields` the CellData names (6 values per element): any of
  // "stress3D6", "strain3D6". Throws ModelError if the file cannot be created.
  VtkhdfWriter(const std::string& file, const std::vector<std::string>& fields,
               const std::vector<std::string>& cell_fields = {}, bool single_precision = false);
  ~VtkhdfWriter();
  VtkhdfWriter(const VtkhdfWriter&) = delete;
  VtkhdfWriter& operator=(const VtkhdfWriter&) = delete;

  void write_mesh(const VtkhdfMesh& mesh);
  // One recorded step: values[f] holds n * 3 doubles of field f (point-major), then one
  // pointer per cell field with m * 6 doubles (cell-major).
  void write_step(double time, const std::vector<const double*>& values);
  void close();

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace femorax::io
