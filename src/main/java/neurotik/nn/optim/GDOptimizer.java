package neurotik.nn.optim;

import neurotik.tensor.Tensor;

public class GDOptimizer extends Optimizer{
    double learningRate;
    public GDOptimizer(double learningRate){

        this.learningRate=learningRate;
    }

    @Override
    public void update(Tensor Parameter,int epoch) {
        for (int i=0;i<Parameter.rows;i++){
            for(int j=0;j<Parameter.cols;j++){
                Parameter.data[i][j]+=-learningRate*Parameter.gradients[i][j];
            }
        }
    }
}
