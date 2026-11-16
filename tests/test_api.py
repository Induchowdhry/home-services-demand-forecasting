"""Flask integration tests. Run after training with: pytest -q"""
import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING']=True
    with app.test_client() as client: yield client

def test_health(client):
    response=client.get('/health'); assert response.status_code==200; assert response.json['status']=='ok'

def test_city_service_catalog(client):
    response=client.get('/cities'); assert response.status_code==200
    assert 'Bangalore' in response.json['cities']; assert 'AC Repair' in response.json['services']

def test_forecast_returns_seven_days(client):
    response=client.post('/forecast',json={'city':'Bangalore','service_type':'AC Repair'})
    assert response.status_code==200
    body=response.json; assert body['forecast_horizon_days']==7; assert len(body['forecast'])==7
    assert all(row['predicted_bookings']>=0 for row in body['forecast'])

def test_missing_parameters(client):
    response=client.post('/forecast',json={'city':'Bangalore'}); assert response.status_code==400

def test_unknown_city(client):
    response=client.post('/forecast',json={'city':'Atlantis','service_type':'AC Repair'}); assert response.status_code==404
