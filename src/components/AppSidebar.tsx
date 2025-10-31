import { useState } from "react";
import { BarChart3, Route, Plane, Upload } from "lucide-react";
import { NavLink, useLocation } from "react-router-dom";

import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarTrigger,
  useSidebar,
} from "@/components/ui/sidebar";

const navigationItems = [
  { 
    title: "Data Analysis", 
    url: "/", 
    icon: BarChart3,
    description: "CSV data insights"
  },
  { 
    title: "Path Visualizer", 
    url: "/path-visualizer", 
    icon: Route,
    description: "3D flight simulation"
  },
];

export function AppSidebar() {
  const { state } = useSidebar();
  const location = useLocation();
  const currentPath = location.pathname;
  const collapsed = state === "collapsed";

  const isActive = (path: string) => currentPath === path;
  const isExpanded = navigationItems.some((item) => isActive(item.url));
  
  const getNavClasses = (active: boolean) =>
    active 
      ? "bg-primary text-primary-foreground font-medium" 
      : "hover:bg-sidebar-accent/50 text-sidebar-foreground";

  return (
    <Sidebar
      className={`${collapsed ? "w-16" : "w-64"} transition-all duration-300`}
      collapsible="icon"
    >
      <SidebarContent className="bg-sidebar border-r border-sidebar-border">
        {/* Header */}
        <div className="p-4 border-b border-sidebar-border">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-primary rounded-lg">
              <Plane className="h-5 w-5 text-primary-foreground" />
            </div>
            {!collapsed && (
              <div>
                <h1 className="font-bold text-lg text-sidebar-foreground">FlightScope</h1>
                <p className="text-xs text-sidebar-foreground/70">Aviation Analytics</p>
              </div>
            )}
          </div>
        </div>

        <SidebarGroup className="px-3">
          <SidebarGroupLabel className="text-sidebar-foreground/70 font-medium mb-2">
            Navigation
          </SidebarGroupLabel>
          
          <SidebarGroupContent>
            <SidebarMenu className="space-y-2">
              {navigationItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton asChild>
                    <NavLink 
                      to={item.url} 
                      className={`${getNavClasses(isActive(item.url))} flex items-center gap-3 px-3 py-3 rounded-lg transition-colors min-h-[3rem]`}
                    >
                      <item.icon className="h-5 w-5 flex-shrink-0" />
                      {!collapsed && (
                        <div className="flex flex-col justify-center min-w-0 flex-1">
                          <span className="font-medium text-sm truncate">{item.title}</span>
                          <span className="text-xs opacity-70 truncate">{item.description}</span>
                        </div>
                      )}
                    </NavLink>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        {/* Flight Status Indicator */}
        {!collapsed && (
          <div className="mt-auto p-4 border-t border-sidebar-border">
            <div className="aviation-card p-3 space-y-2">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-sm font-medium text-sidebar-foreground">System Online</span>
              </div>
              <p className="text-xs text-sidebar-foreground/70 leading-relaxed">
                Ready for flight data analysis
              </p>
            </div>
          </div>
        )}
      </SidebarContent>
    </Sidebar>
  );
}