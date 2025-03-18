#include "LinearRegression.h"

namespace LR {


	MultipleLR::MultipleLR() :
		iteration(), learning_rate()
	{
		slope = NULL;
		RMSEs = NULL;
		intercept = 0.0;
	}

	void MultipleLR::fit( double** X,  double* target, int size, int features, int iteration, double learning_rate) {
		// save iteration and learning rate
		this->intercept = iteration;
		this->learning_rate = learning_rate;
		// save iteration
		this->iteration = iteration;
		this->learning_rate = learning_rate;

		if (this->slope != NULL) {
			std::printf("Every model can be used only once!!!\n");

			return ;
		}

		// initialize slope array
		slope = new double[features];
		RMSEs = new double[iteration];
		for (int j = 0; j < features; ++j) {
			slope[j] = 0.0;
		}

		for (int iter = 0; iter < iteration; ++iter) {
			/*
			Each Iteration
			*/
			double MSE = 0.0;
			double* slopePartialDerivative = new double[features];
			double interceptPartialDerivative = 0.0;

			for (int j = 0; j < features; ++j) {
				slopePartialDerivative[j] = 0.0;
			}

			for (int i = 0; i < size; ++i) {
				/*
				Each Data point
				*/
				double hypothsis = intercept;

				for (int j = 0; j < features; ++j) {
					hypothsis += slope[j] * X[i][j];
				}

				double error = target[i] - hypothsis;
				MSE += error * error;
				
				// calculate slopePartialDerivative and interceptPartialDerivative
				for (int j = 0; j < features; ++j) {
					slopePartialDerivative[j] += error * X[i][j];
				}
				interceptPartialDerivative += error;
			}

			// calculate MSE and save it
			MSE /= size;
			this->RMSEs[iter] = std::sqrt(MSE);

			for (int j = 0; j < features; ++j) {
				slopePartialDerivative[j] = slopePartialDerivative[j] / size * -1;
			}
			interceptPartialDerivative = interceptPartialDerivative / size * -1;

			// update slope and intercept
			for (int j = 0; j < features; ++j) {
				this->slope[j] = this->slope[j] - learning_rate * slopePartialDerivative[j];
			}
			this->intercept = this->intercept - learning_rate * interceptPartialDerivative;

			// clear the dynamic array 
			delete[] slopePartialDerivative;
		}

		for (int j = 0; j < features; ++j) {
			this->parameterSlope.push_back(this->slope[j]);
		}
	}

	std::vector<double> MultipleLR::predict( double** X, int size, int features) {
		std::vector<double> y_predict;
		for (int i = 0; i < size; ++i) {
			double value = this->intercept;
			for (int j = 0; j < features; ++j) {
				value += X[i][j] * this->slope[j];
			}
			y_predict.push_back(value);
		}
		return y_predict;
	}

	MultipleLR::~MultipleLR() {
		delete slope;
		delete RMSEs;
	}

	double MultipleLR::Intercept() {
		return this->intercept;
	}

	std::vector<double> MultipleLR::Slope() {

		return parameterSlope;
	}

	double MultipleLR::r_square(double* target, double* predict, int size, int features) {
		double square = 0.0;
		double mean = 0.0;
		double sum = 0.0;
		double sumofmean = 0.0;

		// calculate the mean
		for (int i = 0; i < size; ++i) {
			sum += target[i];
		}
		mean = sum / (double)size;

		for (int i = 0; i < size; ++i) {
			square += (target[i] - predict[i]) * (target[i] - predict[i]);
			sumofmean += (target[i] - mean) * (target[i] - mean);
		}

		return 1.0 - square / sumofmean;
	}

	double MultipleLR::getRMSE() {
		
		return this->RMSEs[this->iteration - 1];
	}
}