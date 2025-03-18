#include <vector>
#include <cstdio>
#include <cmath>

namespace LR {


	class MultipleLR {
	private:
		double* slope;  // slope
		double intercept;  // intercept
		std::vector<double> parameterSlope;  // use for export
		double* RMSEs;
		int iteration;
		double learning_rate;
		
	public:
		MultipleLR();
		MultipleLR(const MultipleLR&) = delete;
		MultipleLR& operator= (const MultipleLR&) = delete;
		// train model 
		void fit( double** X,  double* target, int size, int features, int iteration, double learning_rate);
		// predict values
		std::vector<double> predict( double** X, int size, int features);
		// R square
		double r_square(double* target, double* predict, int size, int features);
		~MultipleLR();

		// retrieve slope and intercept
		double Intercept();
		std::vector<double> Slope();
		// retrive last MSE
		double getRMSE();
	};
}
