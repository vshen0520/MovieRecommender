from django.shortcuts import render, redirect
from chatting.models import Room, Message
from django.http import HttpResponse, JsonResponse

# Create your views here.
def home(request):
    return render(request, 'home.html')


def room(request, room):
    username = request.GET.get('username')
    room_details = Room.objects.get(name=room)
    return render(request, 'room.html', {
        'username': username,
        'room': room,
        'room_details': room_details
    })
    
    
def checkview(request):
    room = request.POST['room_name']
    username = request.POST['username']
    
    if Room.objects.filter(name=room).exists():
        # if the room already exists, go to that room
        return redirect('/'+room+'/?username='+username)
    else:
        # if the room does not exist, create a new room
        new_room = Room.objects.create(name=room)
        new_room.save()
        return redirect('/'+room+'/?username='+username)

# def returnHome(request):
#     return redirect('/')


def send(request):
    message = request.POST['message']
    username = request.POST['username']
    room_id = request.POST['room_id']
    
    new_message = Message.objects.create(value=message, user=username, room=room_id)
    new_message.save()
    
    return HttpResponse('Message sent successfully')

def getReplies(request):
    message = request.POST['message']
    username = request.POST['username']
    room_id = request.POST['room_id']
    
    new_message = Message.objects.create(value=message, user=username, room=room_id)
    new_message.save()
        
    return JsonResponse({"system_reply": system_reply})


def getMessages(request, room):
    room_details = Room.objects.get(name=room)
    
    messages = Message.objects.filter(room=room_details.id)
    
    return JsonResponse({"messages": list(messages.values())})


def getSingleMessage(request, room):
    if request.method == 'GET':
        # get the lastest message of user
        usr_message = request.GET.get('usr_message')
        # whatever function to get the lastest message of user for further processing
        # ...
        
    return JsonResponse({"system_reply": system_reply})


def systemReply(request):
    if request.method == "POST":
        # get the lastest message of user
        system_reply = request.POST
       
    return JsonResponse({"system_reply": system_reply})