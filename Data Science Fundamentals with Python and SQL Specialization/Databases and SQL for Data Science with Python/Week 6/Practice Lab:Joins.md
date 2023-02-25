
## Problem 1
List the case number, type of crime and community area for all crimes in community area number 18.
```sql
select L.ID, L.CASE_NUMBER, L.PRIMARY_TYPE, L.COMMUNITY_AREA_NUMBER, R.COMMUNITY_AREA_NUMBER, R.COMMUNITY_AREA_NAME
FROM chicago_crime_data L
JOIN chicago_census_data R
On L.COMMUNITY_AREA_NUMBER = R.COMMUNITY_AREA_NUMBER
WHERE L.COMMUNITY_AREA_NUMBER = 18
```

## Problem 2
List all crimes that took place at a school. Include case number, crime type and community name.
```sql
select L.ID, L.CASE_NUMBER, L.PRIMARY_TYPE, L.location_description, R.COMMUNITY_AREA_NAME
FROM chicago_crime_data L
left join  chicago_census_data R
On L.COMMUNITY_AREA_NUMBER = R.COMMUNITY_AREA_NUMBER
where L.location_description like '%SCHOOL%';
```

## Problem 3
For the communities of Oakland, Armour Square, Edgewater and CHICAGO list the associated community_area_numbers and the case_numbers.
```sql
select L.ID, L.CASE_NUMBER, L.PRIMARY_TYPE, L.location_description,R.COMMUNITY_AREA_NUMBER, R.COMMUNITY_AREA_NAME
FROM chicago_crime_data L
full outer join  chicago_census_data R
On L.COMMUNITY_AREA_NUMBER = R.COMMUNITY_AREA_NUMBER
where R.COMMUNITY_AREA_NAME IN ('Oakland', 'Armour Square', 'Edgewater');
```