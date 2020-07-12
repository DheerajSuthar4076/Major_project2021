import random
from django.contrib.auth.decorators import login_required, permission_required
from django.core.exceptions import PermissionDenied
from django.shortcuts import get_object_or_404, render
from django.utils.decorators import method_decorator
from django.views.generic import DetailView, ListView, TemplateView
from django.views.generic.edit import FormView
from .forms import QuestionForm
from .models import Quiz, Category, Progress, Sitting, Question, Subscription
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages

from django.shortcuts import render,redirect
from .models import Student, Contact, Coordinator, School_register
from django.contrib import messages
from django.core.mail import send_mail
from django.conf import settings
from django.contrib.auth import authenticate, login, logout
from django.views.generic import View, UpdateView
from django.urls import reverse_lazy
from django.contrib.auth.models import User
from django.contrib.sites.shortcuts import get_current_site
from django.utils.encoding import force_bytes, force_text
from django.utils.http import urlsafe_base64_encode,urlsafe_base64_decode
from django.template.loader import render_to_string
from quiz.tokens import account_activation_token
from django.contrib.auth.models import User
from mcq.models import MCQQuestion,Answer

my_list =[]
my_answers= []
my_ques= []
#and user.email_confirmed
# def loginhandle(request):
#     if(request.method=='POST'):
#         username=request.POST['email']
#         password=request.POST['pass']
#         user=authenticate(username=username, password=password)
#         print(user)
#         if(user):
#             if(user.is_active ):
#                 login(request,user)
#                 messages.success(request,"Successfully logged in")
#                 return render(request,"loginhandle.html")

#         else:
#             messages.error(request,"User is not registered")

#     return render(request,"loginhandle.html")


# def logouthandle(request):
# 	logout(request)
# 	messages.success(request,"Successfully logged out")
# 	return redirect('/')

#def home(request):
#    return render(request,"home.html")

def subscribe(request):
    if(request.method=='POST'):
        student = request.user
        sub=Student.objects.get(pk=student.id)
        mathsolym=request.POST.get('mathsolym', False)
        scienceolym=request.POST.get('scienceolym',False)
        englisholym=request.POST.get('englisholym', False)
        reasoningolym=request.POST.get('reasoningolym', False)
        cyberolym=request.POST.get('cyberolym', False)
        internationalspell=request.POST.get('internationalspell', False)
        if(sub.mathsolym==False):
            sub.mathsolym=mathsolym
        if(sub.scienceolym==False):
            sub.scienceolym=scienceolym
        if(sub.englisholym==False):
            sub.englisholym=englisholym
        if(sub.reasoningolym==False):
            sub.reasoningolym=reasoningolym
        if(sub.cyberolym==False):
            sub.cyberolym= cyberolym
        if(sub.internationalspell==False):
            sub.internationalspell=internationalspell
        sub.save(update_fields=['mathsolym','scienceolym','englisholym','reasoningolym','cyberolym','internationalspell'])
        


    return render(request,"subscriptions.html")
    

def examdates(request):
    return render(request,"examdates.html")

def faqs(request):
    return render(request,"faqs.html")


def contact(request):

    if(request.method=='POST'):

        name=request.POST['name']
        email=request.POST['email']
        phonenum=request.POST['phonenum']
        message=request.POST['message']
        coordinate=Contact(name=name,email=email,phonenum=phonenum,message=message)

        coordinate.save()

    return render(request,"contact.html")


def coordinator(request):
    if(request.method=='POST'):
        name=request.POST['name']
        email=request.POST['email']
        phonenum=request.POST['phonenum']
        message=request.POST['message']
        coordinate=Coordinator(name=name,email=email,phonenum=phonenum,message=message)
        coordinate.save()
    return render(request,"coordinator.html")


def register_school(request):
    if(request.method=='POST'):
        button=request.POST['button']
        name=request.POST['name']
        email=request.POST['email']
        country=request.POST['country']
        contact=request.POST['contact']
        school_name=request.POST['school_name']
        school_address=request.POST['school_address']
        school_city=request.POST['school_city']
        school_pincode=request.POST['school_pincode']
        school_website=request.POST['school_website']
        school_email=request.POST['school_email']
        pname=request.POST['pname']
        pmobile=request.POST['pmobile']
        exammode=request.POST['exammode']
        if(School_register.objects.filter(school_email=school_email).exists()):
            messages.warning(request,"Email already exists")
        else:
            school_r=School_register(type=button, name=name, email=email,country=country,contact=contact, school_name=school_name,school_address=school_address, school_city=school_city, school_pincode=school_pincode,school_website=school_website,school_email=school_email, pname=pname, pmobile=pmobile, exammode=exammode)
            school_r.save()
            messages.success(request, 'School Registered successful')


    return render(request,"applyschool.html")



def register(request):
    if(request.method=='POST'):

        ref_code = request.POST.get('ref_code', '000')
        # print(ref_code, "ref_Code")
        first_name=request.POST['first_name']
        last_name=request.POST['last_name']
        username= request.POST['username']
        parent_name = request.POST['parent_name']
        dob = request.POST['dob']
        country =request.POST['country']
        address = request.POST['address']
        street= request.POST['street']
        state=request.POST['state']
        school=request.POST['school']
        school_state = request.POST['school_state']
        school_address=request.POST['school_address']
        school_city=request.POST['school_city']
        pincode = request.POST['pincode']
        number=request.POST['number']
        email = request.POST['email']
        password =request.POST['password']
        standard = request.POST['standard']

        if(Student.objects.filter(email=email).exists()):
            messages.warning(request,"Email already exists")
        else:
            if(Student.objects.filter(username=username).exists()):
                messages.warning(request,"Username already exists")
            else:
                student_context=Student(ref_code = ref_code ,first_name=first_name,username=username, last_name=last_name,parent_name = parent_name,dob = dob,country= country,address= address,street=street,state=state,school=school,school_state= school_state,school_address= school_address,school_city= school_city,pincode=pincode,number=number,email=email,standard= standard)
                student_context.set_password(password)
                student_context.save()
                messages.success(request, 'Registration successful')
                print("okkkkkkk")


                # current_site = get_current_site(request)
                # subject = 'Activate Your MySite Account'
                # message = render_to_string('account_activation_email.html', {
                # 'user': context,
                # 'domain': current_site.domain,
                # 'uid': urlsafe_base64_encode(force_bytes(context.pk)),
                # 'token': account_activation_token.make_token(context),
                # })
                # send_mail(subject, message,settings.EMAIL_HOST_USER,email,fail_silently=False,)


                # messages.success(request, ('Please Confirm your email to complete registration.'))
                return render(request,"login.html")





    return render(request,'applyindividual.html')



class ActivateAccount(View):


    def get(self, request, uidb64, token, *args, **kwargs):
            try:
                uid = force_text(urlsafe_base64_decode(uidb64))
                user = Student.objects.get(pk=uid)
            except (TypeError, ValueError, OverflowError, Student.DoesNotExist):
                user = None

            if user is not None and account_activation_token.check_token(user, token):
                user.is_active = True
                user.email_confirmed = True
                user.save()
                login(request, user)
                messages.success(request, ('Your account have been confirmed.'))
                return redirect('/')
            else:
                messages.warning(request, ('The confirmation link was invalid, possibly because it has already been used.'))
                return redirect('/')



class QuizMarkerMixin(object):
    @method_decorator(login_required)
    @method_decorator(permission_required('quiz.view_sittings'))
    def dispatch(self, *args, **kwargs):
        return super(QuizMarkerMixin, self).dispatch(*args, **kwargs)


class SittingFilterTitleMixin(object):
    def get_queryset(self):
        queryset = super(SittingFilterTitleMixin, self).get_queryset()
        quiz_filter = self.request.GET.get('quiz_filter')
        if quiz_filter:
            queryset = queryset.filter(quiz__title__icontains=quiz_filter)

        return queryset


class QuizListView(ListView):
    model = Quiz
    # @login_required
    def get_queryset(self):
        queryset = super(QuizListView, self).get_queryset()
        return queryset.filter(draft=False)
        


class QuizDetailView(DetailView):
    model = Quiz
    slug_field = 'url'

    def get(self, request, *args, **kwargs):
        self.object = self.get_object()

        if self.object.draft and not request.user.has_perm('quiz.change_quiz'):
            raise PermissionDenied

        context = self.get_context_data(object=self.object)
        return self.render_to_response(context)


class CategoriesListView(ListView):
    model = Category


class ViewQuizListByCategory(ListView):
    model = Quiz
    template_name = 'view_quiz_category.html'

    def dispatch(self, request, *args, **kwargs):
        self.category = get_object_or_404(
            Category,
            category=self.kwargs['category_name']
        )

        return super(ViewQuizListByCategory, self).\
            dispatch(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        context = super(ViewQuizListByCategory, self)\
            .get_context_data(**kwargs)

        context['category'] = self.category
        return context

    def get_queryset(self):
        queryset = super(ViewQuizListByCategory, self).get_queryset()
        return queryset.filter(category=self.category, draft=False)


class QuizUserProgressView(TemplateView):
    template_name = 'progress.html'

    @method_decorator(login_required)
    def dispatch(self, request, *args, **kwargs):
        return super(QuizUserProgressView, self)\
            .dispatch(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        context = super(QuizUserProgressView, self).get_context_data(**kwargs)
        progress, c = Progress.objects.get_or_create(user=self.request.user)
        context['cat_scores'] = progress.list_all_cat_scores
        context['exams'] = progress.show_exams()
        return context


class QuizMarkingList(QuizMarkerMixin, SittingFilterTitleMixin, ListView):
    model = Sitting

    def get_queryset(self):
        queryset = super(QuizMarkingList, self).get_queryset()\
                                               .filter(complete=True)

        user_filter = self.request.GET.get('user_filter')
        if user_filter:
            queryset = queryset.filter(user__username__icontains=user_filter)

        return queryset

    class Meta:
        pass


class QuizMarkingDetail(QuizMarkerMixin, DetailView):
    model = Sitting

    def post(self, request, *args, **kwargs):
        sitting = self.get_object()

        q_to_toggle = request.POST.get('qid', None)
        if q_to_toggle:
            q = Question.objects.get_subclass(id=int(q_to_toggle))
            if int(q_to_toggle) in sitting.get_incorrect_questions:
                sitting.remove_incorrect_question(q)
            else:
                sitting.add_incorrect_question(q)

        return self.get(request)

    def get_context_data(self, **kwargs):
        context = super(QuizMarkingDetail, self).get_context_data(**kwargs)
        context['questions'] =\
            context['sitting'].get_questions(with_answers=True)
        return context


class QuizTake(TemplateView):
    template_name = 'question.html'


    def dispatch(self, request, *args, **kwargs):
        self.quiz = get_object_or_404(Quiz, url=self.kwargs['quiz_name'])


        if self.quiz.draft and not request.user.has_perm('quiz.change_quiz'):
            raise PermissionDenied

        self.logged_in_user = self.request.user.is_authenticated

        if self.logged_in_user:
            self.sitting = Sitting.objects.user_sitting(request.user,self.quiz)

            allquestion = self.quiz.get_questions()

            self.progress = self.sitting.progress()
            print(allquestion)
            global bro
            progress, c = Progress.objects.get_or_create(user=self.request.user)
            if(request.method=='POST'):

                bro = 0
                for ques in allquestion:
                    # if(request.POST==None):
                    #     self.question = ques
                    #     answer = Answer.objects.get(id=guess)
                    #     self.sitting.add_incorrect_question(self.question)
                    #     progress.update_score(self.question, 0, 1)
                    # else:
                    guess =request.POST.get('%s'%ques.id,00)
                    print(guess)
                    self.question = ques
                    if(guess==00):
                        print("*************************************")
                        # self.sitting.add_incorrect_question(self.question)
                        # progress.update_score(self.question, 0, 1)
                    else:
                        answer = Answer.objects.get(id=guess)
                        print(answer)
                        print("*************************************")

                        if answer.correct is True:
                            bro = bro+1
                            self.sitting.add_to_score(1)
                            progress.update_score(self.question, 1, 1)
                        else:
                            self.sitting.add_incorrect_question(self.question)
                            progress.update_score(self.question, 0, 1)

                    self.previous = {}

                    self.sitting.add_user_answer(self.question, guess)

                print(bro)
                return self.final_result_user()

        return super(QuizTake, self).dispatch(request, *args, **kwargs)

    # def get_form(self, form_class=QuestionForm):
    #     if self.logged_in_user:
    #         self.question = self.sitting.get_first_question()
    #         self.progress = self.sitting.progress()
    #     return form_class(**self.get_form_kwargs())

    # def get_form_kwargs(self):
    #     kwargs = super(QuizTake, self).get_form_kwargs()

    #     return dict(kwargs, question=self.question ,my_ques = my_ques)


    answers = []
    def get_context_data(self, **kwargs):
        context = super(QuizTake, self).get_context_data(**kwargs)
        # print(self.quiz.get_questions())
        allquestion = self.quiz.get_questions()
        my_answers=[]
        d=[]
        q = {
                'question': '',
                'answers': []
            }
        for ques in allquestion:
            # q = {}
            my_answers=[]
            q = {
                'question': '',
                'answers': []
            }
            q['question'] = ques
            ques_c = MCQQuestion.objects.get(id = int(ques.id))
            ques_choices = [x for x in ques_c.get_answers_list()]
            # q[ques.content]=ques_choices
            # print(ques_choices)
            for choice in ques_choices:
                my_answers.append(choice)
            # print(q)
                # q[ques.content]=choice[1]
            q['answers']=my_answers
            # print(q)
                # q.answers.append(choice[1])
            d.append(q)

        context['my_ques'] = d
        if hasattr(self, 'previous'):
            context['previous'] = self.previous
        if hasattr(self, 'progress'):
            context['progress'] = self.progress
        context['q_length'] = len(d)
        return context


    def final_result_user(self):
        per = (bro/self.sitting.get_max_score)*100
        results = {
            'quiz': self.quiz,
            'score': bro,
            'max_score': self.sitting.get_max_score,
            'percent': per,
            'sitting': self.sitting,

            #'previous': self.previous,
        }

        self.sitting.mark_quiz_complete()

        if self.quiz.exam_paper is False:
            self.sitting.delete()

        return render(self.request, 'result.html', results)




def index(request):
    return render(request, 'index.html', {})


def login_user(request):

    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(username=username, password=password)
        if user is not None:
            login(request, user)
            messages.success(request, 'You have successfully logged in')
            return render(request, 'index.html')
        else:
            messages.error(request,"User is not registered")

    return render(request, 'login.html')


def logout_user(request):
    logout(request)
    messages.success(request, 'You have been logged out!')
    print('logout function working')
    return redirect('loginhandle')












# def loginhandle(request):
#     if(request.method=='POST'):
#         username=request.POST['email']
#         password=request.POST['pass']
#         user=authenticate(username=username, password=password)
#         print(user)
#         if(user):
#             if(user.is_active ):
#                 login(request,user)
#                 messages.success(request,"Successfully logged in")
#                 return render(request,"loginhandle.html")

#         else:
#             messages.error(request,"User is not registered")

#     return render(request,"loginhandle.html")


# def logouthandle(request):
# 	logout(request)
# 	messages.success(request,"Successfully logged out")
# 	return redirect('/')