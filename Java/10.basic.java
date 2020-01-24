// 조건문 
int a = 10;
int b = 20;

if(a < b) {
    System.out.println("a is bigger than b");
} else if(a > b) {
    System.out.println("b is bigger than a");
}
else {
    System.out.println("a equals b");
}

// for문
System.out.print("Input number: ");
Scanner scanner = new Scanner(System.in);
int inputNum = scanner.nextInt();

for (int i = 1; i < 10; i++) {
    System.out.printf("%d * %d = %d\n", inputNum, i, (inputNum * i));
}

// while문
System.out.print("Input number: ");
int num = scanner.nextInt();
int i = 1
while (i < 10) {
    System.out.printf("%d * %d = %d\n", inputNum, i, (inputNum * i));
}
 
do {
    System.out.printf("%d * %d = %d\n", inputNum, i, (inputNum * i));
} while (false);

// OOP
package prjBasic;

public class Pet {

    public String name;
    public int age;

    pubic Pet() {
        System.out.println("Nana Class Construction")
    }

    public void run() {
        System.out.println("Run!");
    }
}

Nana dog1 = new Pet();
dog1.run()


// Method 
public void myFunc(int i, boolean b, double d, String s) {
    System.out.println("some application~ ")
}


