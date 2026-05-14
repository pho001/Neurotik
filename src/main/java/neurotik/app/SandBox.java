package neurotik.app;

import tensor.CompileMode;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

public class SandBox {
    public SandBox(){
        double [][] x0=new double[][]{{1}};
        double [][] x1=new double[][]{{2}};
        double [][] x2=new double[][]{{3}};

        double [][] y0=new double[][]{{2}};
        double [][] y1=new double[][]{{3}};
        double [][] y2=new double[][]{{4}};

        //double [][] h=new double[][]{{0}};

        double [][] wf=new double[][]{{0.1},{0.2}};
        double [][] wi=new double[][]{{0.3},{0.4}};
        double [][] wo=new double [][]{{0.7},{0.5}};
        double [][] wc=new double [][]{{0.8},{0.6}};
        double [][] h0=new double [][]{{0}};
        double [][] ct=new double [][]{{0}};
        double [][] y=new double[][]{{0.1},{0.5}};


        Tensor X0=new Tensor(x0, List.of(), "x0", DataType.FLOAT64);
        Tensor X1=new Tensor(x1, List.of(), "x1", DataType.FLOAT64);
        Tensor X2=new Tensor(x2, List.of(), "x2", DataType.FLOAT64);

        Tensor Y0=new Tensor(y0, List.of(), "y1", DataType.FLOAT64);
        Tensor Y1=new Tensor(y1, List.of(), "y2", DataType.FLOAT64);
        Tensor Y2=new Tensor(y2, List.of(), "y3", DataType.FLOAT64);

        Tensor Wf=new Tensor(wf, List.of(), "wf", DataType.FLOAT64).trainableParameter();
        Tensor Wi=new Tensor(wi, List.of(), "wi", DataType.FLOAT64).trainableParameter();
        Tensor Wo=new Tensor(wo, List.of(), "wo", DataType.FLOAT64).trainableParameter();
        Tensor Wc=new Tensor(wc, List.of(), "wc", DataType.FLOAT64).trainableParameter();

        Tensor H0=new Tensor(h0, List.of(), "h0", DataType.FLOAT64);
        Tensor C0=new Tensor(ct, List.of(), "c0", DataType.FLOAT64);

        /*
        Tensor A=new Tensor(wf,new HashSet<>(),"A");
        Tensor B=new Tensor(wi,new HashSet<>(),"B");
        Tensor C=A.hadamard(B);
        Tensor Y=new Tensor(y,new HashSet<>(),"Y");
        */
        //Tensor L=C.mse(Y);
        //L.backward();

        //step 0
        Tensor Z0=Tensor.concat(1, X0, H0);
        Tensor forgetGate0=Z0.matmul(Wf).sigmoid();
        Tensor inputGate0=Z0.matmul(Wi).sigmoid();
        Tensor candidateCellState0=Z0.matmul(Wc).tanh();

        Tensor C1=forgetGate0.mul(C0).add(inputGate0.mul(candidateCellState0));
        Tensor outputGate=Z0.matmul(Wo).sigmoid();
        Tensor H1=outputGate.mul(C1.tanh());

        Tensor L0=H1.sub(Y0).pow(2).mean();





        //step 1

        Tensor Z1=Tensor.concat(1, X1, H1);
        Tensor forgetGate1=Z1.matmul(Wf).sigmoid();
        Tensor inputGate1=Z1.matmul(Wi).sigmoid();
        Tensor candidateCellState1=Z1.matmul(Wc).tanh();

        Tensor C2=forgetGate1.mul(C1).add(inputGate1.mul(candidateCellState1));
        Tensor outputGate1=Z1.matmul(Wo).sigmoid();
        Tensor H2=outputGate1.mul(C2.tanh());

        Tensor L1=H2.sub(Y1).pow(2).mean();



        //step 2
        Tensor Z2=Tensor.concat(1, X2, H2);
        Tensor forgetGate2=Z2.matmul(Wf).sigmoid();
        Tensor inputGate2=Z2.matmul(Wi).sigmoid();
        Tensor candidateCellState2=Z2.matmul(Wc).tanh();

        Tensor C3=forgetGate2.mul(C2).add(inputGate2.mul(candidateCellState2));
        Tensor outputGate2=Z2.matmul(Wo).sigmoid();
        Tensor H3=outputGate2.mul(C3.tanh());

        Tensor L2=H3.sub(Y2).pow(2).mean();
        Tensor Loss=L0.add(L1.add(L2));
        Loss.compute(CompileMode.TRAINING);



    }
}
