package neurotik.app;

import neurotik.tensor.Tensor;

import java.util.HashSet;

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


        Tensor X0=new Tensor(x0,new HashSet<>(),"x0");
        Tensor X1=new Tensor(x1,new HashSet<>(),"x1");
        Tensor X2=new Tensor(x2,new HashSet<>(),"x2");

        Tensor Y0=new Tensor(y0,new HashSet<>(),"y1");
        Tensor Y1=new Tensor(y1,new HashSet<>(),"y2");
        Tensor Y2=new Tensor(y2,new HashSet<>(),"y3");

        Tensor Wf=new Tensor(wf,new HashSet<>(),"wf");
        Tensor Wi=new Tensor(wi,new HashSet<>(),"wi");
        Tensor Wo=new Tensor(wo,new HashSet<>(),"wo");
        Tensor Wc=new Tensor(wc,new HashSet<>(),"wc");

        Tensor H0=new Tensor(h0,new HashSet<>(),"h0");
        Tensor C0=new Tensor(ct,new HashSet<>(),"c0");

        /*
        Tensor A=new Tensor(wf,new HashSet<>(),"A");
        Tensor B=new Tensor(wi,new HashSet<>(),"B");
        Tensor C=A.hadamard(B);
        Tensor Y=new Tensor(y,new HashSet<>(),"Y");
        */
        //Tensor L=C.mse(Y);
        //L.backward();

        //step 0
        Tensor Z0=X0.concatRight(H0);
        Tensor forgetGate0=Z0.mul(Wf).sigmoid();
        Tensor inputGate0=Z0.mul(Wi).sigmoid();
        Tensor candidateCellState0=Z0.mul(Wc).tanh();

        Tensor C1=forgetGate0.hadamard(C0).add(inputGate0.hadamard(candidateCellState0));
        Tensor outputGate=Z0.mul(Wo).sigmoid();
        Tensor H1=outputGate.hadamard(C1.tanh());

        Tensor L0=H1.mse(Y0);





        //step 1

        Tensor Z1=X1.concatRight(H1);
        Tensor forgetGate1=Z1.mul(Wf).sigmoid();
        Tensor inputGate1=Z1.mul(Wi).sigmoid();
        Tensor candidateCellState1=Z1.mul(Wc).tanh();

        Tensor C2=forgetGate1.hadamard(C1).add(inputGate1.hadamard(candidateCellState1));
        Tensor outputGate1=Z1.mul(Wo).sigmoid();
        Tensor H2=outputGate1.hadamard(C2.tanh());

        Tensor L1=H2.mse(Y1);



        //step 2
        Tensor Z2=X2.concatRight(H2);
        Tensor forgetGate2=Z2.mul(Wf).sigmoid();
        Tensor inputGate2=Z2.mul(Wi).sigmoid();
        Tensor candidateCellState2=Z2.mul(Wc).tanh();

        Tensor C3=forgetGate2.hadamard(C2).add(inputGate2.hadamard(candidateCellState2));
        Tensor outputGate2=Z2.mul(Wo).sigmoid();
        Tensor H3=outputGate2.hadamard(C3.tanh());

        Tensor L2=H3.mse(Y2);
        Tensor Loss=L0.add(L1.add(L2));
        Loss.backward();



    }
}
