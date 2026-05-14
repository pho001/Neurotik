package neurotik.app;

import neurotik.encoding.Encoder;
import neurotik.tensor.MathHelper;
import tensor.Tensor;


import java.util.*;

public class Main {
    public static void main(String[] args) {
        //new SandBox();
        //Demos.runRnnTextDemo();
        //Demos.runRnnNumbersDemo();
        //Demos.runMLPTextDemo();
        //Demos.runMLPNumbersDemo();
        //Demos.runLSTMTextDemo();
        //Demos.runLSTMNumbersDemo();
        //Demos.runGRUNumbersDemo();
        Demos.runGRUTextDemo();



    }


    public static void update(double descent, HashSet<Tensor> params) {
        for (Tensor p : params) {
            Tensor gradient = p.getGradient();
            if (gradient == null) {
                continue;
            }
            double[] data = p.toDoubleArrayCopy();
            double[] grad = gradient.toDoubleArrayCopy();
            for (int i = 0; i < data.length; i++) {
                data[i] += -descent * grad[i];
            }
            p.setData(data);
        }
    }


    public static void generate(int samples, int contextLength, Tensor EMB, Tensor W1, Encoder encoder) {
        /*
        for (int i=0;i<samples;i++){
            String output="";
            String context="";
            Tensor outFlat=null;
            Random random=new Random();
            Tensor [] input=new Tensor[contextLength-1];
            for (int j = 0; j < contextLength-1; j++) {
                context += ".";
            }



            while (true){
                input=encoder.encode(context);

                Tensor P.mul(W1).softMax();
                int iChar=MathHelper.sampleFromMultinomial(1,P.data[0],random)[0];
                char nextChar=encoder.decode(iChar);
                output=output+nextChar;
                if (iChar==0){
                    break;
                }
                context = context.substring(1) + nextChar;

            }
            System.out.println(output);


        }

         */

    }

    public static double[] sinWave(int points, double step){

        double[] sinValues = new double[points];
        for (int i = 0; i < points; i++) {
            sinValues[i] = Math.sin(i*step);
        }

        return sinValues;
    }




}
