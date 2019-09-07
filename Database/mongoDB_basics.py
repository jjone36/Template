import requests
from pymongo import MongoClient

# Connect to the server
client = MongoClient()

# Create a database
db = client['nobel']
db = client.nobel
# db_list = client.list_database_names()
# cl_list = db.list_collections_names()

# Populate the database
dataset = {'prizes' : 'http://api.nobelprize.org/v1/prize.json',
           'laureates' : 'http://api.nobelprize.org/v1/laureate.json'}
for name in list(dataset.keys()):
    url = dataset[name]
    response = requests.get(url)
    doc = response.json()[name]
    db[name].insert_many(doc)

n_prizes = db.prizes.count_documents({})
prize_1 = db.prizes.find_one({})
prize_fields = list(prize_1.keys())

# Count the number of documents in a prize collection
len(cl['name'] for cl in db.prizes.find())


#################################################
############ Filter
# $ne, $in, $nin, $lt, $gt, $gte, #exists
stm = {'born' : {'$lt' : '1900'}, 'bornCountry' : 'Germany'}
stm = {'bornCountry' : {"$in": ['USA', 'Canada', 'Mexico']}}
stm = {'prizes.affiliations.name' : 'University of California'}
db.laureates.count_documents(stm)

stm = {'prizes.2' : {'$exists' : True}}
set(db.laureates.distinct('category', stm))

# $elemMatch
stm = {"prizes": {'$elemMatch': {
        'category': {'$nin': ["physics", "chemistry", "medicine"]},
        "share": "1",
        "year": {'$gte': "1945"},
        }}}
db.laureates.count_documents(stm)

# $regex
from bson.regex import Regex
stm = {'firstname' : {'$regex' : '^G'}, 'surmane' : {'$regex' : '^S'}}
stm = {'firstname' : Regex('^G'), 'surmane' : Regex('^S')}


############ Projection
stm = {'gender' : 'org'}, ['bornCountry', 'firstname', '_id' : 0]
db.laureates.find(stm)[:3]
db.laureates.find(stm).limit(3)

stm = {'firstname' : Regex('^G'), 'surname' : Regex('^S'), ['firstname', 'surname']}
for cl in db.laureates.find(stm):
    fullname = " ".join(cl['firstname'], cl['surname'])
    print(fullname)


############ sorting
stm = {'year' : {'$gt' : 1900, '$lt' : 2000}, ['category']}
print([doc['year'] for doc in db.prizes.find(stm, sort= [('year', -1), ('category', 1)])][:5])

db.prizes.find(stm).sort('year', -1).limit(5)


############ indexing
db.prizes.create_index([('category', 1), ('year', -1)])
db.prizes.index_information()
# quering plans
db.prizes.find(stm).explain()

cate_list = db.prizes.distinct('category')
outputs = []
for cate in sorted(cate_list):
    stm = {'category' : cate, 'laureates.share' : '1'}
    doc = db.prizes.find_one(stm, sort = [('year', -1)])
    outputs.append('{category}: {year}'.format(**doc))


#################################################
############ Aggregation
# https://docs.mongodb.com/manual/reference/operator/
# {"$operator" : "$field_name"}
from collections import OrderedDict
db.laureate.aggregate([
    {'$match' : {'bornCountry' : 'USA'}},
    {'$project' : ['year', 'bornCountry']},
    {'$order' : OrderedDict('year', -1)}
    ]).next()

# $count, $size, $sum, $group
pipe = [{'$match' : {'gender' : 'org'}},
        {'$project' : {'n_prizes' : {'$size' : '$prizes'}}},
        {'$group' : {'_id' = None, 'n_prizes_total' : {'$sum' : '$n_prizes'}}}]
db.laureates.aggregate(pipe)

# pipeline
stm = {'year' : '1945'}
cate_in_1945 = sorted(set(db.prizes.distinct('category', stm)))
pipe = [
    {'$match' : {'category' : {'$in' : cate_in_1945}}},
    {'$project' : {'category' : 1, 'year' : 1}},
    # Group the categories for each year
    {'$group' : {'_id' : '$year', 'categories_in_year' : {'$addToSet' : '$category'}}},
    # Get the list of categories not in cate_in_1945
    {'$project' : {'missing_cate' : {'$setDifference' : [cate_in_1945, '$categories_in_year']}}},
    # Filter only the year missing_cate_in_year exists
    {"$match": {"missing_cate.0": {"$exists": True}}},
    {'$sort' : OrderedDict([('_id', -1)])}
    ]

for doc in db.prizes.aggregate(pipe):
    missing = ', '.join(sorted(doc['missing_cate']))
    print("{} : {}".format(doc['_id'], missing))


# $unwind, $lookup
pipe = [
    {'$unwind': "$laureates"},
    {"$lookup": {
        "from": "laureates",
        "foreignField": "id",
        "localField": "laureates.id",
        "as": "laureate_bios"
                }},

    {"$unwind": '$laureate_bios'},
    {"$project": {"category": 1, "bornCountry": "$laureate_bios.bornCountry"}},

    # Collect bornCountry values associated with each prize category
    {"$group": {'_id': "$category", "bornCountries": {"$addToSet": "$bornCountry"}}},
    # Project out the size of each category's (set of) bornCountries
    {"$project": {"category": 1, "nBornCountries": {"$size": '$bornCountries'}}},
    {"$sort": {"nBornCountries": -1}},
        ]


# $addFields
