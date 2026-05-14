package neurotik.nn.init;

public class InitializerFactory {
    public static Initializer randomInit(){
        return new RandomInit();
    }
    public static Initializer normalInit(double stdDev,double mean){
        return new NormalInit(stdDev,mean);
    }

    public static Initializer kaimingInit(){
        return new KaimingInit();
    }

    public static Initializer uniformInit(double upper, double lower){
        return new UniformInit(upper, lower);
    }

    public static Initializer xavierInit(){
        return new XavierInit();
    }


}
