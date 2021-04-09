"""
.. module:: docs_test
   :synopsis: All endpoints of the Teacher API are defined here
.. moduleauthor:: Rich Yap <github.com/richyap13>
"""

from flask import request, jsonify, abort, make_response, Blueprint
from teacherAPI.database import db_session
from teacherAPI.models import Teacher
from sqlalchemy import exc
import os

teacher_api = Blueprint('teacher_api', __name__)


@teacher_api.route('/', methods=['GET'])
def index():
    """
        **Get List of Teachers**
        This function allows users to get a list of teachers and the subjects they're teaching.
        :return: teacher's information in json and http status code
        - Example::
              curl -X GET http://localhost:5000/ -H 'cache-control: no-cache' -H 'content-type: application/json'
        - Expected Success Response::
            HTTP Status Code: 200
            {
                "Teachers": [
                    {
                        "id": 1,
                        "name": "Jane Vargas",
                        "subject": "Science"
                    },
                    {
                        "id": 2,
                        "name": "John Doe",
                        "subject": "Math"
                    },
                    {
                        "id": 3,
                        "name": "Jenny Lisa",
                        "subject": "English"
                    }
                ]
            }
    """
    teachers = Teacher.query.all()
    return make_response(jsonify(Teachers=[Teacher.serialize() for Teacher in teachers]), 200)


@teacher_api.route('/', methods=['POST'])
def create_teacher():
    """
        **Create Teacher Record**
        This function allows user to create(post) a teacher record.
        :return: teacher's information added by the user in json and http status code
        - Example::
            curl -X POST http://localhost:5000/ -H 'cache-control: no-cache' -H 'content-type: application/json' \
            -d '{
                "name": "Mary Rose",
                "subject": "Biology"
            }'
        - Expected Success Response::
            HTTP Status Code: 201
            {
                "name": "Mary Rose",
                "subject": "Biology"
            }
        - Expected Fail Response::
            HTTP Status Code: 400
            {'error': 'Duplicate teacher name'}
    """
    if not request.json:
        abort(400)
    content = request.get_json()
    if type(content['name']) != str:
        return make_response(jsonify({'error':'Teacher name should be a string'}), 400)
    try:
        teacher_temp = Teacher(name=content['name'],
                               subject=content['subject'])
        db_session.add(teacher_temp)
        db_session.commit()
        return jsonify(content), 201
    except exc.IntegrityError as e:
        return make_response(jsonify({'error': 'Duplicate teacher name'}), 400)


@teacher_api.route('/<int:teacher_id>', methods=['GET'])
def get_teacher(teacher_id):
    """
        **Get information of a specific teacher**
        This function allows user to get a specific teacher's information through their teacher_id.
        :param teacher_id: id of the teacher
        :type teacher_id: int
        :return: teacher's information accessed by user in json and http status code
        - Example::
            curl -X GET http://127.0.0.1:5000/1 -H 'cache-control: no-cache' -H 'content-type: application/json'
        - Expected Success Response::
            HTTP Status Code: 200
            {
                "Teacher": {
                    "id": 1,
                    "name": "Jane Vargas",
                    "subject": "Science"
                }
            }
        - Expected Fail Response::
            HTTP Status Code: 404
            {'error': 'Not found'}
    """
    teachers = Teacher.query.all()
    teacher = [teacher for teacher in teachers if teacher.id == teacher_id]
    if len(teacher) == 0:
        not_found()
    return make_response(jsonify(Teacher=Teacher.serialize(teacher[0])), 200)


@teacher_api.route('/<int:teacher_id>', methods=['PUT'])
def update_teacher(teacher_id):
    """
        **Update Information of a Specific Teacher Record**
        This function allows user to update a specific teacher's information through their teacher_id.
        :param teacher_id: id of the teacher
        :type teacher_id: int
        :return: teacher's information updated by user in json and http status code
        - Example::
            curl -X PUT http://localhost:5000/1 -H 'cache-control: no-cache' -H 'content-type: application/json' \
            -d '{
                "name": "Jane Cruz",
                "subject": "Science"
            }'
        - Expected Success Response::
            HTTP Status Code: 200
            {
                "name": "Jane Cruz",
                "subject": "Science"
            }
        - Expected Fail Response::
            HTTP Status Code: 404
            {'error': 'Not found'}
            or
            HTTP Status Code: 404
            {'error': 'Duplicate teacher name'}
    """
    teachers = Teacher.query.all()
    teacher = [teacher for teacher in teachers if teacher.id == teacher_id]
    if len(teacher) == 0:
        not_found()
    if 'name' in request.json and type(request.json['name']) != str:
        return make_response(jsonify({'error': 'Teacher name not a string'}), 400)
    if 'subject' in request.json and type(request.json['subject']) != str:
        return make_response(jsonify({'error': 'Subject not a string'}), 400)
    content = request.get_json()
    # updating the requested teacher record
    try:
        queried_teacher = Teacher.query.get(teacher_id)
        queried_teacher.name = content['name']
        queried_teacher.subject = content['subject']
        db_session.commit()
        return make_response(jsonify(content), 200)
    except exc.IntegrityError as e:
        return make_response(jsonify({'error': 'Duplicate teacher name'}), 400)


@teacher_api.route('/<int:teacher_id>', methods=['DELETE'])
def delete_teacher(teacher_id):
    """
        **Delete Teacher Record**
        This function allows user to delete a teacher record.
        :param teacher_id: id of the teacher
        :type teacher_id: int
        :return: delete status in json and http status code
        - Example::
            curl -X DELETE http://127.0.0.1:5000/4 -H 'cache-control: no-cache' -H 'content-type: application/json'
        - Expected Success Response::
            HTTP Status Code: 200
            {
                "Delete": true
            }
        - Expected Fail Response::
            HTTP Status Code: 404
            {'error': 'Not found'}
    """
    teachers = Teacher.query.all()
    teacher = [teacher for teacher in teachers if teacher.id == teacher_id]
    if len(teacher) == 0:
        not_found()
    Teacher.query.filter_by(id=teacher_id).delete()
    db_session.commit()
    return make_response(jsonify({'Delete': True}), 200)


@teacher_api.route('/search', methods=['POST'])
def search():
    """
        **Search Teacher Records**
        This function allows user to search for teacher/s through substring search of teachers' names.
        :return: searched teachers in json and http status code
        - Example::
            curl -X POST  http://localhost:5000/search -H 'cache-control: no-cache' -H 'content-type: application/json' \
            -d '{
                "value": "J"
            }'
        - Expected Success Response::
            HTTP Status Code: 200
            {
                "Teachers": [
                    {
                        "id": 1,
                        "name": "Jane Cruz",
                        "subject": "Science"
                    },
                    {
                        "id": 2,
                        "name": "John Doe",
                        "subject": "Math"
                    },
                    {
                        "id": 3,
                        "name": "Jenny Lisa",
                        "subject": "English"
                    }
                ]
            }
    """
    if not request.json:
        abort(400)
    if 'value' in request.json and type(request.json['value']) is not str:
        abort(400)
    content = request.get_json()
    teachers = Teacher.query.filter(Teacher.name.like('%' + content['value']+ '%'))
    return make_response(jsonify(Teachers=[Teacher.serialize() for Teacher in teachers]), 200)


# Act as an error handler when a page is not found
@teacher_api.errorhandler(404)
def not_found():
    """
        **Error handler**
        This function returns a not found error in json when called.
        :return: not found error in json
    """
    return make_response(jsonify({'error': 'Not found'}), 404)