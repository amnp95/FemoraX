// Reference driver for constraint handling: runs OpenSees' TransformationConstraintHandler
// and PlainNumberer on a node/fix/equalDOF case and writes each nodal DOF's equation.
//
//   dof_numbering <case.txt> <out.txt>
//
// Case format: tests/reference/make_constraint_cases.py. equalDOF is built exactly as the
// OpenSees command does (MP_Constraint.cpp, OPS_EqualDOF: identity Ccr, rcDOF for both
// nodes); fix becomes one SP_Constraint per DOF with value 0.
//
// Output (driver_common.hpp format): one row per node, columns tag ndf e0 e1 e2 ..., where
// e<d> is the equation of nodal DOF d, -1 if fixed. For a node with a TransformationDOF_Group
// the equation is read through the transformation matrix T (nodal DOF i -> modID column j
// with T(i, j) = 1), as the element assembly sees it.

#include "driver_common.hpp"

#include <AnalysisModel.h>
#include <DOF_Group.h>
#include <Domain.h>
#include <ID.h>
#include <LoadControl.h>
#include <MP_Constraint.h>
#include <Matrix.h>
#include <Node.h>
#include <PlainNumberer.h>
#include <SP_Constraint.h>
#include <TransformationConstraintHandler.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: dof_numbering <case.txt> <out.txt>\n");
    return 1;
  }
  Domain domain;
  std::vector<int> tags;
  int max_ndf = 0;
  std::ifstream in(argv[1]);
  for (std::string line; std::getline(in, line);) {
    std::istringstream w(line);
    std::string kind;
    w >> kind;
    if (kind == "node") {
      int tag, ndf;
      double x, y, z;
      w >> tag >> ndf >> x >> y >> z;
      domain.addNode(new Node(tag, ndf, x, y, z));
      tags.push_back(tag);
      max_ndf = std::max(max_ndf, ndf);
    } else if (kind == "fix") {
      int tag, dof;
      w >> tag >> dof;
      domain.addSP_Constraint(new SP_Constraint(tag, dof - 1, 0.0, true));
    } else if (kind == "equal") {
      int r, c;
      w >> r >> c;
      std::vector<int> dofs;
      for (int d; w >> d;) dofs.push_back(d - 1);
      const int n = static_cast<int>(dofs.size());
      Matrix Ccr(n, n);
      ID rcDOF(n);
      for (int i = 0; i < n; ++i) {
        rcDOF(i) = dofs[i];
        Ccr(i, i) = 1.0;
      }
      domain.addMP_Constraint(new MP_Constraint(r, c, Ccr, rcDOF, rcDOF));
    }
  }

  AnalysisModel model;
  TransformationConstraintHandler handler;
  LoadControl integrator(1.0, 1, 1.0, 1.0);
  handler.setLinks(domain, model, integrator);
  model.setLinks(domain, handler);
  if (handler.handle() < 0) {
    std::fprintf(stderr, "handler failed\n");
    return 1;
  }
  PlainNumberer numberer;
  numberer.setLinks(model);
  if (numberer.numberDOF() < 0) {
    std::fprintf(stderr, "numberer failed\n");
    return 1;
  }
  handler.doneNumberingDOF();

  std::vector<std::string> cols = {"tag", "ndf"};
  for (int d = 0; d < max_ndf; ++d) cols.push_back("e" + std::to_string(d));
  refdrv::Writer out(argv[2], {std::string("case ") + argv[1]}, cols);
  for (int tag : tags) {
    Node* node = domain.getNode(tag);
    const int ndf = node->getNumberDOF();
    DOF_Group* g = node->getDOF_GroupPtr();
    const ID& id = g->getID();
    Matrix* T = g->getT();
    std::vector<double> row = {static_cast<double>(tag), static_cast<double>(ndf)};
    for (int i = 0; i < max_ndf; ++i) {
      int eq = -9;  // padding beyond ndf
      if (i < ndf) {
        if (T == nullptr) {
          eq = id(i);
        } else {
          eq = -1;  // a nodal DOF with no column in T contributes to no equation
          for (int j = 0; j < T->noCols(); ++j)
            if ((*T)(i, j) != 0.0) eq = id(j);
        }
        if (eq < 0) eq = -1;
      }
      row.push_back(eq);
    }
    out.row(row);
  }
  return 0;
}
