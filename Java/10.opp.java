// ParentClass.java
package tutorials;

public class ParentClass {
    
    int account = 100;
    
    public ParentClass() {
        System.out.println("Parent Class Constructed!");
    }
    public void greeting() {
        System.out.println("Hi There~");
    }
}

// FirstChildClass.java
public class FirstChildClass extends ParentClass {
    
    int account = 200;
    
    public FirstChildClass() {
        System.out.println("First Child Class Constructed!");
    }

    @Override
    public void greeting() {
        System.out.println("Hi I'm a first child");
    }

    public void getAccout() {
        System.out.println("Parent Account : " + super.account);
        System.out.println("First Child Account : " + this.account);
    }
}

// SecondChildClass.java
public class SecondChildClass extends ParentClass {
    
    int account = 300;
    
    public SecondChildClass() {
        System.out.println("Second Child Class Constructed!");
    }
    
    @Override
    public void greeting() {
        System.out.println("Hi I'm a second child");
    }
}

// MainClass.java
public class MainClass {
    public static void main(String[] arg) {
        
        ParentClass fch = new FirstChildClass();
        ParentClass sch = new SecondChildClass();
        
        ParentClass[] pArr = new ParentClass[2];
        pArr[0] = fch;
        pArr[1] = sch;

        fch.greeting();
        sch.greeting();

        fch.getAccout();

    }
}

