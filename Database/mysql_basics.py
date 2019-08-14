import mysql.connector

mydb = mysql.connector.connect(
            host = 'localhost',
            user = 'root',
            passed = '123password'
            # database = 'testdb'
            )

c = mydb.cursor()

c.execute("SHOW DATABASES")
c.execute("CREATE DATABASES testdb")

for db in c:
    print(db)

c.execute("CREATE TABLE students (name VARCHAR(255), age INTEGER(10))")
c.execute("SHOW TABLE")
for tb in c:
    print(tb)

######## Populating a table
stm = "INSERT INTO students (name, age) VALUES (%s, %s)"
ex = [('Jean', '24'),
      ('Jackman', '52'),
      ('Will', '35')]

c.executemany(stm, ex)
mydb.commit()

######## Selecting data
stm = "SELECT * FROM students WHERE name = %s"
c.execute(stm, 'Wi%')
results = c.fetchall()

for result in results:
    print(result)

######## Updating data
stm = "UPDATE students SET age = %s WEHRE name = %s"
c.execute(stm, 13, 'Jean')
mydb.commit()

######## Delete data
stm = "DELETE FROM students WHERE name = %s"
c.execute(stm, 'Jean')
mydb.commit()

######## Drop a table
stm = "DROP TABLE IF EXISTS students"
c.exectue(stm)
mydb.commit()
