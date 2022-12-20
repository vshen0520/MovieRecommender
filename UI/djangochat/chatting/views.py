from django.shortcuts import render, redirect
from chatting.models import Room, Message
from django.http import HttpResponse, JsonResponse
from chatting.preload import MovieREC

def home(request):
    # homepage
    return render(request, 'home.html')


def room(request, room):
    # chatroom
    username = request.GET.get('username')
    room_details = Room.objects.get(name=room)
    return render(request, 'room.html', {
        'username': username,
        'room': room,
        'room_details': room_details
    })
    
    
def checkview(request):
    # check if the room exists
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


def send(request):
    # send message to database
    message = request.POST['message']
    username = request.POST['username']
    room_id = request.POST['room_id']
    
    new_message = Message.objects.create(value=message, user=username, room=room_id)
    new_message.save()
    
    return HttpResponse('Message sent successfully')


def getMessages(request, room):
    # get whole messages from the room (for real-time realization) 
    room_details = Room.objects.get(name=room)
    
    messages = Message.objects.filter(room=room_details.id)
    
    return JsonResponse({"messages": list(messages.values())})


def systemReply(request, room):
    # get latest reply from user, 
    # go over NLP and recommend algorith,
    # and return system reply

    # if request.method == 'GET':
    #     # get the lastest message of user
    #     usr_message = request.GET.get('usr_message')
        # IF NOT WORKING, TRY THIS:
        room_details = Room.objects.get(name=room)
        # usr_message = Message.objects.filter(room=room_details.id).last().value
        messageList = [""]
        for messageDict in list(Message.objects.filter(room=room_details.id).values()):
            messageList.append(messageDict['value'])

        #  whatever function to get the lastest message of user for further processing
        # ...
        # eg. sys_message = myRecommend(usr_message)
        # 
        
        # for testing purpose, just return the lastest message of user
        # sys_message = usr_message
        sys_message = MovieREC.conversation_recommend(messageList)
        
        return JsonResponse({"sys_message": sys_message})
    # else:
    #     return HttpResponse('No Message sent')


