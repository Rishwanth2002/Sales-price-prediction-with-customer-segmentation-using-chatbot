import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { FileuploadComponent } from './fileupload/fileupload.component';
import { LoginComponent } from './login/login.component';
import { ResultComponent } from './result/result.component';
import { ClusterComponent } from './cluster/cluster.component';
import { ChatbotComponent } from './chatbot/chatbot.component';
import { FeedbackComponent } from './feedback/feedback.component';
import { ThankyouComponent } from './thankyou/thankyou.component';

const routes: Routes = [
  {
    path: '',component:LoginComponent
  },{
    path: 'fileupload',component:FileuploadComponent
  },{
    path: 'result',component:ResultComponent
  },{
    path: 'cluster',component:ClusterComponent
  },{
    path: 'chatbot',component:ChatbotComponent
  },{
    path: 'feedback',component:FeedbackComponent
  },{
    path: 'thankyou',component:ThankyouComponent
  }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule],
  declarations: [
    ResultComponent
  ]
})
export class AppRoutingModule { }
