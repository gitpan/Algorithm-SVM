#include "bindings.h"

SVM::SVM(int st, int kt, double d, double g, double c0, double C, double nu,
	 double e) {

  // Default parameter settings.
  param.svm_type = st;
  param.kernel_type = kt;
  param.degree = d;
  param.gamma = g;
  param.coef0 = c0;
  param.nu = nu;
  param.cache_size = 40;
  param.C = 1;
  param.eps = 1e-3;
  param.p = e;
  param.shrinking = 1;
  param.nr_weight = 0;
  param.weight_label = NULL;
  param.weight = NULL;
  param.probability = 0;
  
  x_space = NULL;
  model   = NULL;
  prob    = NULL;

  randomized = 0;
}

void SVM::addDataSet(DataSet *ds) {

  if(ds != NULL) dataset.push_back(ds);
}


void SVM::clearDataSet() {

  dataset.clear();
}

int SVM::train(int retrain) {
  const char *error;
  int nelem = 0;

  // Free any old model we have.
  if(model != NULL) {
    svm_destroy_model(model);
    model = NULL;
  }

  if(retrain) {
    if(prob == NULL) return 0;
    model = svm_train(prob, &param);
    return 1;
  }

  if(x_space != NULL) free(x_space);
  if(prob != NULL) free(prob);

  x_space = NULL;
  model   = NULL;
  prob    = NULL;

  // Allocate memory for the problem struct.
  if((prob = (struct svm_problem *)malloc(sizeof(struct svm_problem))) == NULL) return 0;

  prob->l = dataset.size();

  // Allocate memory for the labels/nodes.
  prob->y = (double *)malloc(sizeof(double) * prob->l);
  prob->x = (struct svm_node **)malloc(sizeof(struct svm_node) * prob->l);

  if((prob->y == NULL) || (prob->x == NULL)) {
    if(prob->y != NULL) free(prob->y);
    free(prob);
    return 0;
  }

  // Check for errors with the parameters.
  error = svm_check_parameter(prob, &param);
  if(error) return 0;

  // Figure out the total number of elements.
  for(int i = 0; i < prob->l; i++) nelem += dataset[i]->attributes.size() + 1;
  x_space = (struct svm_node *)malloc(sizeof(struct svm_node) * nelem);

  if(x_space == NULL) {
    free(prob->y);
    free(prob->x);
    free(prob);
    return 0;
  }

  // Munge the datasets into the format that libsvm expects.
  int n = 0, maxi = 0; 
  for(int i = 0; i < prob->l; i++) {
    prob->x[i] = &x_space[n];
    prob->y[i] = dataset[i]->getLabel();

    map<int,double>::iterator j;
    for(j = dataset[i]->attributes.begin(); j != dataset[i]->attributes.end(); j++) {
      x_space[n].index = (*j).first;
      x_space[n].value = (*j).second;
      n++;
    }

    if(n >= 1 && x_space[n-1].index > maxi) maxi = x_space[n-1].index;

    x_space[n++].index = -1;
  }

  if(param.gamma == 0) param.gamma = 1.0/maxi;

  model = svm_train(prob, &param);
  
  return 1;
}

double SVM::predict(DataSet *ds) {
  struct svm_node *node;
  double pred;

  if(ds == NULL) return 0;
  
  node = (struct svm_node *)malloc(sizeof(struct svm_node) * (ds->attributes.size() + 1));
  if(node == NULL) return 0;
    
  map<int,double>::iterator i;
  int j = 0;
  for(i = ds->attributes.begin(); i != ds->attributes.end(); i++) {
    node[j].index = (*i).first;
    node[j].value = (*i).second;
    j++;
  }
  node[j].index = -1;

  pred = svm_predict(model, node);

  free(node);

  return pred;
}

int SVM::saveModel(char *filename) {

  if((model == NULL) || (filename == NULL)) {
    return 0;
  } else {
    return ! svm_save_model(filename, model);
  }
}

int SVM::loadModel(char *filename) {
  struct svm_model *tmodel;

  if(filename == NULL) return 0;

  if(x_space != NULL) {
    free(x_space);
    x_space = NULL;
  }

  if(model != NULL) {
    svm_destroy_model(model);
    model = NULL;
  }

  if((tmodel = svm_load_model(filename)) != NULL) {
    model = tmodel;
    return 1;
  }

  return 0;
}

double SVM::crossValidate(int nfolds) {
  double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
  double total_error = 0;
  int total_correct = 0;
  int i;

  if(! prob) return 0;

  if(! randomized) {
    // random shuffle
    for(i=0;i<prob->l;i++) {
      int j = i+rand()%(prob->l-i);
      struct svm_node *tx;
      double ty;

      tx = prob->x[i];
      prob->x[i] = prob->x[j];
      prob->x[j] = tx;

      ty = prob->y[i];
      prob->y[i] = prob->y[j];
      prob->y[j] = ty;
    }

    randomized = 1;
  }

  for(i=0;i<nfolds;i++) {
    int begin = i*prob->l/nfolds;
    int end = (i+1)*prob->l/nfolds;
    int j,k;
    struct svm_problem subprob;

    subprob.l = prob->l-(end-begin);
    subprob.x = (struct svm_node**)malloc(sizeof(struct svm_node)*subprob.l);
    subprob.y = (double *)malloc(sizeof(double)*subprob.l);

    k=0;
    for(j=0;j<begin;j++) {
      subprob.x[k] = prob->x[j];
      subprob.y[k] = prob->y[j];
      ++k;
    }

    for(j=end;j<prob->l;j++) {
      subprob.x[k] = prob->x[j];
      subprob.y[k] = prob->y[j];
      ++k;
    }

    if(param.svm_type == EPSILON_SVR || param.svm_type == NU_SVR) {
      struct svm_model *submodel = svm_train(&subprob,&param);
      double error = 0;
      for(j=begin;j<end;j++) {
	double v = svm_predict(submodel,prob->x[j]);
	double y = prob->y[j];
	error += (v-y)*(v-y);
	sumv += v;
	sumy += y;
	sumvv += v*v;
	sumyy += y*y;
	sumvy += v*y;
      }
      svm_destroy_model(submodel);
      // cout << "Mean squared error = %g\n", error/(end-begin));
      total_error += error;			
    } else {
      struct svm_model *submodel = svm_train(&subprob,&param);

      int correct = 0;
      for(j=begin;j<end;j++) {
	double v = svm_predict(submodel,prob->x[j]);
	if(v == prob->y[j]) ++correct;
      }
      svm_destroy_model(submodel);
      //cout << "Accuracy = " << 100.0*correct/(end-begin) << " (" <<
      //correct << "/" << (end-begin) << endl;
      total_correct += correct;
    }

    free(subprob.x);
    free(subprob.y);
  }		
  if(param.svm_type == EPSILON_SVR || param.svm_type == NU_SVR) {
    // printf("Cross Validation Mean squared error = %g\n",total_error/prob->l);
    // printf("Cross Validation Squared correlation coefficient = %g\n",
    return ((prob->l*sumvy-sumv*sumy)*(prob->l*sumvy-sumv*sumy))/
      ((prob->l*sumvv-sumv*sumv)*(prob->l*sumyy-sumy*sumy));

    //);
  } else {
    return 100.0*total_correct/prob->l;
    //cout << "Cross Validation Accuracy = " << 100.0*total_correct/prob->l << endl;
  }
}

int SVM::getNRClass() {

  if(model == NULL) {
    return 0;
  } else {
    return svm_get_nr_class(model);
  }
}

int SVM::getLabels(int* label) {
    if(model == NULL) {
	return 0;
    } else {
	svm_get_labels(model, label);
	return 1;
    }
}

double SVM::getSVRProbability() {

  if((model == NULL) || (svm_check_probability_model(model))) {
    return 0;
  } else {
    return svm_get_svr_probability(model);
  }
}

int SVM::checkProbabilityModel() {

  if(model == NULL) {
    return 0;
  } else {
    return svm_check_probability_model(model);
  }
}


SVM::~SVM() {

  if(x_space != NULL) free(x_space);
  if(model != NULL) svm_destroy_model(model);
  if(prob != NULL) free(prob);
}
