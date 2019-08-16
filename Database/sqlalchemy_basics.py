####################### SQL with python #######################
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('sqlite:///Datacamp.sqlite')
table_names = engine.table_names()
print(table_names)

con = engine.connect()
rs = con.execute('SELECT * FROM Album')
con.close()

# fetchall()
df = pd.DataFrame(rs.fetchall())
print(df.head())

# fetchmany()
with engine.connect() as con:
    rs = con.execute('SELECT LastName, Title FROM Employee')
    df = pd.DataFrame(rs.fetchmany(3))
    df.columns = rs.keys()

# One line!
df = pd.read_sql_query('SELECT LastName, Title FROM Employee', engine)

############# statement
from sqlalchemy import select
print(engine.table_names())

# Reflect census table via engine: census
census = Table('census', metadata, autoload = True, autoload_with = engine)

# filtering: WHERE
stmt = select([census])
stmt = stmt.where(census.columns.state.in_(OECD))
print(stmt)

for result in con.execute(stmt):
    print(result.age, result.gender)

results = con.execute(stmt).fetchall()
for result in results:
    print(result.age, result.gender, result.state)

# filtering: and_, or_
from sqlalchemy import and_, or_
stmt = select([census]).where(
  and_(census.columns.state == 'New York',
       or_(census.columns.age == 21, census.columns.age == 37))
  )

# ordering
from sqlalchemy import desc
stmt = select([census.columns.state, census.columns.age])
stmt = stmt.order_by('state', desc('age'))

# as
stmt = select((census.columns.pop_2008 - census.columns.pop_2000).label('pop_diff'))

# aggregating: sum, count, distinct
from sqlalchemy import func
stmt = select([func.sum(census.columns.age)])
stmt = select([func.count(census.columns.state.distinct()])
n_distinct_state = con.execute(stmt).scalar()

# grouping
stmt = select([census.columns.state, func.count(census.columns.age)])
stmt = stmt.group_by('state')

age_sum = func.sum(census.columns.age).label('age_sum')
stmt = select([census.columns.state, age_sum]).group_by('state')

result = con.execute(stmt).fetchall()
print(result[:20])

import pandas as pd
df = pd.DataFrame(result)
df.columns = result[0].keys()

# joining
stmt = select([census, account])
stmt = stmt.select_from(
         census.join(account, census.columns.state == account.columns.name))

# updating
stmt = select([census]).where(census.columns.state == 'New York')

stmt2 = update(census).values(census.columns.age == 30).where(census.columns.state == 'New York')
result = con.execute(stmt2)
print(result.rowcount)

# deleting
stmt_del = delect(census).where(census.colums.age >= 80)
