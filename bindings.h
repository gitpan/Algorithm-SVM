#ifndef __BINDINGS_H__
#define __BINDINGS_H__

#include <vector>
#include <map>

#include "libsvm.h"

class DataSet {
  friend class SVM;

 private:
  double label;
  map<int,double> attributes;
 public:
  DataSet(double l) { label = l; }
  void   setLabel(double l) { label = l; }
  double getLabel() { return label; }
  void   setAttribute(int k, double v) { attributes[k] = v; }
  double getAttribute(int k) { return attributes[k]; }
  ~DataSet() { };
};


class SVM {
 public:
  SVM(int st, int kt, double d, double g, double c0, double C, double nu,
      double e);
  void   addDataSet(DataSet *ds);
  int    saveModel(char *filename);
  int    loadModel(char *filename);
  void   clearDataSet();
  int    train(int retrain);
  double predict(DataSet *ds);
  void   setSVMType(int st) { param.svm_type = st; }
  int    getSVMType() { return param.svm_type; }
  void   setKernelType(int kt) { param.kernel_type = kt; }
  int    getKernelType() { return param.kernel_type; }
  void   setGamma(double g) { param.gamma = g; }
  double getGamma() { return param.gamma; }
  void   setDegree(double d) { param.degree = d; }
  double getDegree() { return param.degree; }
  void   setCoef0(double c) { param.coef0 = c; }
  double getCoef0() { return param.coef0; }
  void   setC(double c) { param.C = c; }
  double getC() { return param.C; }
  void   setNu(double n) { param.nu = n; }
  double getNu() { return param.nu; }
  void   setEpsilon(double e) { param.p = e; }
  double getEpsilon() { return param.p; }
  double crossValidate(int nfolds);

  ~SVM();
 private:
  struct svm_parameter param;
  vector<DataSet *> dataset;
  struct svm_problem *prob;
  struct svm_model *model;
  struct svm_node *x_space;
  int randomized;
};

#endif
