a = {
    "count": 6,
    "next": "",
    "previous": "",
    "results": [
        {
            "parent": "",
            "uid": "a85r79ZxP",
            
        },
        {
            "owner": "https://kf.kobotoolbox.org/api/v2/users/hpitchik.json",
            "uid": "aeY85r79ZxP",
            
        }
    ]
}

key_list=a.get('results')
a_key = "uid"

list_of_values = [a_dict[a_key] for a_dict in key_list]
print(list_of_values)