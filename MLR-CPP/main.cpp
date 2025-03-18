// Machine-Learnung.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include "LinearRegression.h"
#include "external/rapidcsv.h"

int main(int argc, char *argv[])
{
	rapidcsv::Document file("./myDataMLR.csv");
	std::vector<double> x1 = file.GetColumn<double>("x1");
	std::vector<double> x2 = file.GetColumn<double>("x2");
	std::vector<double> y = file.GetColumn<double>("y");

	if (x1.size() != x2.size() || x1.size() != y.size()) {
		std::printf("each column must have same number of items\n");
		return -1;
	}

	const int SIZE = x1.size();
	const int FEATURES = 2;

	if ( argc != 3 ) {
		std::printf("Usage: %s 1000 0.01\n", argv[0]);
		return 0;
	}

	int iteration = std::atoi( argv[1] );
	double learning_rate = std::atof( argv[2] );

	

	// data loader
	// independent variables and targets
	double** independences;
	double* targets;
	
	
	independences = new double* [SIZE];
	for (int i = 0; i < SIZE; ++i) {
		independences[i] = new double[FEATURES];
		
	}
	
	targets = new double[SIZE];

	for (int i = 0; i < SIZE; ++i) {
		independences[i][0] = x1[i];
		independences[i][1] = x2[i];
		targets[i] = y[i];
		
	}

	// Model build, train and test

	LR::MultipleLR lrModel{};
	
	const auto start{ std::chrono::steady_clock::now() };  // start time
	lrModel.fit(independences, targets, SIZE, FEATURES, iteration, learning_rate);
	const auto finish{ std::chrono::steady_clock::now() };  // end time
	std::vector <double> y_predict = lrModel.predict(independences, SIZE, FEATURES);
	double r2_score = lrModel.r_square(targets, &y_predict[0], SIZE, FEATURES);
	std::vector<double> slope = lrModel.Slope();
	double intercept = lrModel.Intercept();

	const std::chrono::duration<double> elapsed_seconds{ finish - start };

	printf("y = %.2f x1 + %.2f x2 + %.2f\n", slope[0], slope[1], intercept);
	printf("RMSE: %.6lf\n", lrModel.getRMSE());
	printf("Time Durtion: {%.10f}\n", elapsed_seconds.count());
	printf("The R2 Score: %.4f", r2_score);

	
	for (int i = 0; i < SIZE; ++i) {
		delete independences[i];
		independences[i] = NULL;
	}
	delete[] independences;
	independences = NULL;
	delete[] targets;
	targets = NULL;

}
