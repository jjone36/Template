// 1. Object 
// var obj = new Object();
// var obj = {};

var obj = {
    name : "sally",
    age : 3,
    friends : ["cony", "brown"],
    family : {mother : "billy"},
    talk : function(){ return "hello"}
};

// obj;
// obj.name;
// obj['name'];
// obj.job;

var key = "name";
// obj.key;
obj[key];

obj.name = "billy";
obj['name'] = "billy";
obj.money = 100;

// 2. Array 
var arr = new Array(3);
var arr = [1, 2, 3];

arr[0];     // 1
arr[5];     // undefined
arr.length;

arr[1] = 5;
arr.push(4);
arr[10] = 10;

// 3. Wrapper object
var a = 123;
a.toString();   // String으로 반환 후 다시 int로 
a;

// 4. (원시형 복사와) 참조형 복사
var obj1 = {name : "sally"}
var obj2 = obj2;
obj2.name;

obj2["name"] = "brown"
obj1.name;

// 5. 객체 비교하기 
var obj1 = {name : "sally"}
var obj2 = {name : "sally"}

obj1 == obj2;   // false

// 6. Nested Object 
var obj1 = {inner: {name: "sally"}}
var obj2 = {inner: obj1.inner}

obj2.inner.name = "brown"
obj1.inner.name

// 7. for in 문
var obj = {
    a : 1,
    b : 2,
    c : 3
};

for (var key in obj) {
    console.log(key, ": ", obj[key])
}

// 8. delete  
var obj = {
    member : 123,
}

delete obj.member;
obj;
delete obj;

// 9. call by value 
var a = 1;
function addOne(n) {
    return n+1;
}
addOne(a);      // 2
a;      // 1

var obj = {};
function addName(obj) {
    obj.name = "sally";
}
addName(obj);   
obj     // {name : "sally"}

// 10. function
function fn(val) {
    return val;
}
fn.name;    // "fn"

fn.inner = 123;

var obj = {myFn: fn};
obj.myFn(2);

var arr = [fn, fn, fn];
arr[0](3);

fn(fn);

// 11. polymorphism
function add(a, b) {
    return a + b;
}

add(1, 2);
add(1);
add(1, 2, 3);

// 12. arguments object 
function fn() {
    return arguments;
}
fn();
fn(1, 2, 3);

Math.max(1, 2, 3, 4, 5, 6, 7, 8)

// 13. this keyword
function fn(name) {
    this.name = name;
    return this;
}
// 전역에서 호출 
fn("sally");    // window
name           // window

// method로 호출
var obj = {method: fn};
obj.method("brown");

// 생성자로 호출 -> 객체로 생성
new fn("moon");

var obj2 = {
    outer: function() {
        return function() {
            this.name = "inner";
            return this;
        }
    }
}
obj2.outer();       // -> function() { ~~~ }
obj2.outer()();     // = function() 따라서 전역으로 호출, 반환값 window 

// 14. call, apply, bind
function fn(age) {
    this.age = age;
}
var obj = {};

fn.call(obj, 10);   // call의 첫번째 인자를 this로 받음
obj; 

fn.bind(obj, 10);   // -> fn function 이지만 
obj.age             // this와 인자를 bind 해줌

// 15. Prototype
function Sally() {
    this.dream = "지구정복!"
}
Sally.prototype     

var sally1 = new Sally();
sally1.__proto__

Sally.prototype.weapon = "gun"
var sally2 = new Sally()
sally2;
sally2.weapon;      // 객체 자신이 가지고 있지 않지만 프로토타입이 가지고 있는 멤버가 참조 가능함 (변경은 불가)
sally2.weapon = "bomb"
Sally.prototype.weapon;

// hasOwnProperty()는 프로토타입 체인으로 동작하지 않음  
sally2.hasOwnProperty("dream");     // -> true
sally2.hasOwnProperty("weapon");    // -> false

function Sally() {
    this.method = function(){};
}
// Sally.prototype.method = function(){};       // 프프로토타입을 사용하면 변경 
var sally1 = new Sally();
var sally2 = new Sally();
sally1.method == sally2.method      // false, 참조값이 다름 

// 16. heritage 
function Sally2(){
    Sally.apply(this);     // -> 상속
    this.mask = "멋짐"
}

var sally = new Sally2();
sally;
Sally2.prototype.__proto__ = Sally.prototype    // 프로토타입 상속 

// Class?? 실제 구현 방식은 프로토타입이고 편하게 객체를 작성할 수 잇도록 마련된 신텍스 슈거임 
class Sally {
    constructor() {
        this.dream = "지구정복";
    }
    method() {
        return this.dream;
    }
}

// 17. Scope 
var a = 1;
function outer() {
    var b = 2;
    function inner() {
        b = 3;
        return a + b;
    }
    return inner();
}

outer();

// 18. hoesting? 
var a = 1;
function fn() {
    a = 2;
    var a = 3;
}

a           // -> 1
fn()        // -> 3


function outer() {
    console.log(inner1, inner2);
    function inner1(){}
    var inner2 = function() {};
}

outer();

// 19. clouser 
var a = 1;
function outer() {
    var a = 2;
    return function inner() {
        return a;
    }
}
var inner = outer();
inner();

var fnList = [];
for (var i = 0; i<3; ++i) {
    fnList[i] = function() {
        return i;
    }
}

var fnList2 = [];
for (var i = 0; i <3; ++i) {
    function out() {
        var j = i;
        return function inner() {
            return j;
        }
    }
    fnList2[i] = out();
}
fnList[0]();
fnList[1]();
fnList[2]();

fnList2[0]();
fnList2[1]();
fnList2[2]();

// 
// 20. Monkey Patch (객체를 선언 시점이 아닌 사용 시점에서 확장하는 것)
var obj1 = {method : function() {this.val = 1}};
var obj2 = {};

obj2.method = obj1.method;
obj2.method();
obj2;


function fn() {
    arguments.forEach = Array.prototype.forEach;
    arguments.forEach(function (v){console.log(v)});
}
fn(1, 2, 3);

function fn2() {
    Array.prototype.forEach.call(
        arguments, function (v) {console.log(v)}
    );
}
fn2(1, 2, 3);

// 21. Mixin
var obj1 = {a: 1, b: 2};
var obj2 = {c: 3 ,d: 4};
var mixin = {};

for (var key in obj1) {mixin[key] = obj1[key]};
for (var key in obj2) {mixin[key] = obj2[key]};
mixin;
// var mixin = {...obj1, ...obj2}

// 22. Pattern 
var mixin = {...Array.prototype, ...String.prototype};
function StringArray() {
    Array.call(this);
    String.call(this);
}
StringArray.prototype = mixin;

// duck typing 
var obj1 = {
    val: 123,
    method: function() {
    if (this.val === undefined){
        throw new Error("val 없음");
    }  
    return this.val;
    }
};
var obj2 = {};

// chainable method 
var obj = {
    a: function() {
        console.log("a");
        // return this;
    },
    b: function() {
        console.log("b");
        // return this;
    }
}
obj.a();
obj.b();
// obj.a().b().a().b();


// currying 
function forFn(todo, data) {
    for(var key in data) {
        todo(data[key]);
    }
}
forFn(function(v){console.log(v)},[1, 2, 3])
forFn(function(v){console.log(v)},[4, 5, 6])
// 반복이니깐, 다음처럼~ 

function forFn(todo, data) {
    if (!data) {
        return function(data) {forFn(todo, data)}
    } else {
        for(var key in data) {
            todo(data[key]);
        }
    }
}
var console = forFn(function(v){console.log(v)})
console([1, 2, 3]);
console([4, 5, 6]);


